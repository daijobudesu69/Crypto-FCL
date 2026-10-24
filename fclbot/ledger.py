"""Ledger funding Binance: sumber sinyal FCL, direkam tanpa akses ke Binance.

Setiap siklus (~10 menit) bot membaca predictedFundings HYPE (venue BinPerp): rate yang
sedang terbentuk untuk settlement Binance berikutnya. Snapshot terakhir sebelum settlement
disimpan di state/funding_pending.json; begitu waktu settlement lewat, snapshot itu ditulis
ke state/funding_ledger.csv sebagai rate settlement (source=predicted). Watcher bangun 5 menit
sebelum tiap jam settlement (kelipatan 4 jam UTC), jadi snapshot itu hampir final.

Celah sebelum bot jalan diisi tools/bootstrap_funding.py (source=bootstrap), yang
merekonstruksi funding dari arsip premium index harian Binance. Kalau satu settlement punya
dua sumber, `predicted` yang dipakai.

Sinyal F14 = rata-rata funding per jam selama 14 hari x 336. Jam yang tidak tercatat tidak
diisi; coin butuh cakupan >= min_coverage. Setara jumlah 14 hari kalau cakupan penuh.
"""
from __future__ import annotations

import datetime as dt

from . import store

PENDING = "funding_pending.json"
KEEP_DAYS = 40


def iso(ms: int) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat(timespec="seconds")


def record(snap: dict, now: dt.datetime) -> dict:
    """snap: {coin: {rate, next_ms, interval_h}} dari predicted_fundings(). Return ringkasan."""
    now_ms = int(now.timestamp() * 1000)
    pend = store.load_json(PENDING, {}) or {}
    done = 0
    for coin, p in list(pend.items()):
        new = snap.get(coin)
        passed = now_ms >= p["next_ms"] or (new is not None and new["next_ms"] > p["next_ms"])
        if passed:
            store.append("funding_ledger", {"settle_utc": iso(p["next_ms"]), "settle_ms": p["next_ms"], "coin": coin,
                                            "rate": p["rate"], "interval_h": p["interval_h"], "source": "predicted",
                                            "seen_utc": p["seen_utc"]})
            done += 1
            del pend[coin]
    seen = now.isoformat(timespec="seconds")
    for coin, s in snap.items():
        if s["next_ms"] > now_ms:
            pend[coin] = {"rate": s["rate"], "next_ms": s["next_ms"], "interval_h": s["interval_h"], "seen_utc": seen}
    store.save_json(PENDING, pend)
    return {"settled": done, "pending": len(pend)}


def rows_in_window(now: dt.datetime, days: int) -> list[dict]:
    now_ms = int(now.timestamp() * 1000)
    lo = now_ms - days * 86_400_000
    best: dict[tuple, dict] = {}
    for r in store.read("funding_ledger"):
        t = int(r["settle_ms"])
        if not (lo < t <= now_ms):
            continue
        k = (r["coin"], t)
        if k not in best or (best[k]["source"] == "bootstrap" and r["source"] == "predicted"):
            best[k] = r
    return list(best.values())


def f14(now: dt.datetime, days: int = 14) -> dict:
    """{coin: (signal, coverage)}; signal = rata-rata funding per jam x 24 x days."""
    agg: dict[str, list] = {}
    for r in rows_in_window(now, days):
        a = agg.setdefault(r["coin"], [0.0, 0.0])
        a[0] += float(r["rate"])
        a[1] += float(r["interval_h"])
    hours = days * 24
    return {c: (s / h * hours if h > 0 else None, min(h / hours, 1.0)) for c, (s, h) in agg.items()}


def prune(now: dt.datetime) -> int:
    rows = store.read("funding_ledger")
    lo = int(now.timestamp() * 1000) - KEEP_DAYS * 86_400_000
    keep = [r for r in rows if int(r["settle_ms"]) >= lo]
    if len(keep) < len(rows):
        store.rewrite("funding_ledger", keep)
    return len(rows) - len(keep)
