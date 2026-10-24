"""Buku paper FCL: posisi dalam qty koin, mark-to-market pakai harga mid HYPE.

* mark()     : PnL harga sejak mark terakhir.
* funding()  : funding HYPE per jam yang benar-benar terjadi (fundingHistory), long bayar kalau
               rate > 0. Dihitung atas notional saat ini (pendekatan, selisihnya kecil).
* stops()    : short ditutup kalau mid >= entry x 1,40 (fill + 0,5% slip). Di live nanti ini order
               trigger yang menunggu di exchange; di paper dicek tiap siklus (~10 menit).
* rebalance(): trade ke target. Biaya 0,065%/sisi; selisih < $10 tidak dikirim, kecuali buka/tutup.
"""
from __future__ import annotations

import datetime as dt

from . import store

FILE = "paper.json"


def load(equity0: float) -> dict:
    st = store.load_json(FILE)
    if not st:
        st = {"equity": equity0, "equity0": equity0, "peak": equity0, "positions": {}, "fees_cum": 0.0,
              "funding_cum": 0.0, "price_pnl_cum": 0.0, "banned": [], "last_rebalance_day": None,
              "started_utc": None}
    return st


def save(st: dict) -> None:
    store.save_json(FILE, st)


def mark(st: dict, mids: dict) -> float:
    pnl = 0.0
    for coin, pos in st["positions"].items():
        px = mids.get(coin)
        if px is None:
            continue
        pnl += pos["qty"] * (px - pos["last_px"])
        pos["last_px"] = px
    st["equity"] += pnl
    st["price_pnl_cum"] += pnl
    st["peak"] = max(st["peak"], st["equity"])
    return pnl


def funding(st: dict, info, now: dt.datetime) -> float:
    total = 0.0
    now_ms = int(now.timestamp() * 1000)
    for coin, pos in st["positions"].items():
        since = int(pos.get("funding_ms") or pos["opened_ms"])
        if now_ms - since < 3_600_000:
            continue
        for t, rate in info.funding_history(coin, since + 1, now_ms):
            if t <= since:
                continue
            pay = -pos["qty"] * pos["last_px"] * rate
            total += pay
            since = t
        pos["funding_ms"] = since
    st["equity"] += total
    st["funding_cum"] += total
    return total


def _fill(st, day, coin, qty_delta, px, kind, reason, cost, now):
    notional = abs(qty_delta) * px
    fee = notional * cost
    st["equity"] -= fee
    st["fees_cum"] += fee
    store.append("orders", {"time_utc": now.isoformat(timespec="seconds"), "day": day, "coin": coin,
                            "side": "buy" if qty_delta > 0 else "sell", "kind": kind, "qty": qty_delta, "px": px,
                            "notional": notional, "fee": fee, "reason": reason})


def stops(st: dict, mids: dict, p: dict, cost: float, now: dt.datetime) -> list:
    hit = []
    for coin, pos in list(st["positions"].items()):
        px = mids.get(coin)
        if px is None or pos["qty"] >= 0:
            continue
        trigger = pos["entry_px"] * (1 + p["stop_pct"])
        if px >= trigger:
            fill = px * (1 + p["stop_slip"])
            slip = pos["qty"] * (fill - px)           # qty < 0 -> rugi tambahan
            st["equity"] += slip
            st["price_pnl_cum"] += slip
            _fill(st, now.date().isoformat(), coin, -pos["qty"], fill, "stop", f"stop +{p['stop_pct']:.0%}", cost, now)
            pnl = pos["qty"] * (fill - pos["entry_px"])
            store.append("stops", {"time_utc": now.isoformat(timespec="seconds"), "coin": coin,
                                   "entry_px": pos["entry_px"], "trigger_px": trigger, "fill_px": fill, "pnl": pnl})
            del st["positions"][coin]
            if coin not in st["banned"]:
                st["banned"].append(coin)
            hit.append((coin, pnl))
    return hit


def rebalance(st: dict, weights: dict, mids: dict, cost: float, min_order: float, now: dt.datetime) -> int:
    day = now.date().isoformat()
    eq = st["equity"]
    n = 0
    coins = set(weights) | set(st["positions"])
    # tutup dan kurangi dulu, baru buka (urutan sama dengan rencana live)
    plan = []
    for coin in coins:
        px = mids.get(coin)
        if px is None:
            continue
        pos = st["positions"].get(coin)
        cur = pos["qty"] * px if pos else 0.0
        tgt = weights.get(coin, 0.0) * eq
        if tgt != 0 and abs(tgt) < min_order:
            tgt = 0.0
        flip = pos is not None and tgt != 0 and (tgt > 0) != (pos["qty"] > 0)
        plan.append((0 if (tgt == 0 or abs(tgt) < abs(cur) or flip) else 1, coin, px, pos, cur, tgt, flip))
    for _, coin, px, pos, cur, tgt, flip in sorted(plan):
        ms = int(now.timestamp() * 1000)
        if pos and (tgt == 0 or flip):
            _fill(st, day, coin, -pos["qty"], px, "close", "keluar dari buku" if tgt == 0 else "ganti sisi", cost, now)
            del st["positions"][coin]
            pos, cur = None, 0.0
            n += 1
        if tgt == 0:
            continue
        if pos is None:
            if abs(tgt) < min_order:
                continue
            q = tgt / px
            _fill(st, day, coin, q, px, "open", "masuk buku", cost, now)
            st["positions"][coin] = {"qty": q, "entry_px": px, "last_px": px, "opened_ms": ms, "funding_ms": ms,
                                     "opened_utc": now.isoformat(timespec="seconds")}
            n += 1
        elif abs(tgt - cur) >= min_order:
            dq = (tgt - cur) / px
            _fill(st, day, coin, dq, px, "adjust", "rebalance", cost, now)
            pos["qty"] += dq
            n += 1
    return n


def snapshot(st: dict) -> dict:
    eq = st["equity"]
    long_ = sum(p["qty"] * p["last_px"] for p in st["positions"].values() if p["qty"] > 0)
    short = sum(p["qty"] * p["last_px"] for p in st["positions"].values() if p["qty"] < 0)
    return {"equity": eq, "peak": st["peak"], "dd_pct": (eq / st["peak"] - 1) * 100 if st["peak"] else 0.0,
            "gross": (long_ - short) / eq if eq else 0.0, "net": (long_ + short) / eq if eq else 0.0,
            "n_long": sum(1 for p in st["positions"].values() if p["qty"] > 0),
            "n_short": sum(1 for p in st["positions"].values() if p["qty"] < 0),
            "fees_cum": st["fees_cum"], "funding_cum": st["funding_cum"], "price_pnl_cum": st["price_pnl_cum"]}
