"""Satu siklus FCL: rekam funding -> mark buku paper -> stop -> funding HYPE -> rebalance harian.

Kegagalan sumber funding (predictedFundings HYPE) tidak boleh menghentikan mark dan stop.
Kalau feed BinPerp putus >= FEED_ALERT_H jam, bot memberi alarm dan mengisi ledger dari
arsip premium index Binance (fclbot/bootstrap.py), supaya F14 tetap punya cakupan.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from . import bootstrap, ledger, notify, paper, store, strategy

FEED = "feed.json"
FEED_MIN_SHARE = 0.5      # snapshot sehat = BinPerp ada untuk >= 50% koin coins.csv
FEED_ALERT_H = 8          # tanpa snapshot sehat selama 8 jam (>= 1 settlement tiap koin) -> alarm + arsip
FEED_ALERT_EVERY_H = 6
ARCHIVE_EVERY_H = 6
ARCHIVE_MAX_DAYS = 5      # hari arsip per percobaan, supaya satu siklus tetap < 15 menit
CANDLE_FAIL_MAX = 0.10    # > 10% koin gagal diambil candle-nya -> rebalance ditunda ke siklus berikut
CANDLE_RETRY_H = 4        # ...paling lama 4 jam setelah jadwal, lalu jalan dengan data yang ada


@dataclass
class Ctx:
    cfg: object
    info: object
    ctrl: object
    outbox: object
    now: dt.datetime


def rebalance_due(cfg, now: dt.datetime, last_day: str | None) -> bool:
    h, m = cfg.rebalance_hm
    at = now.replace(hour=h, minute=m, second=0, microsecond=0)
    return now >= at and last_day != now.date().isoformat()


def _hours(a: dt.datetime, b) -> float:
    return (a - dt.datetime.fromisoformat(b)).total_seconds() / 3600


def run(ctx: Ctx) -> dict:
    cfg, info, now = ctx.cfg, ctx.info, ctx.now
    out = {"snapshot": None, "daily": False, "errors": []}
    if ctx.ctrl.mode == "off":
        return out

    snap = None
    try:
        snap = {c: v for c, v in info.predicted_fundings().items() if c in cfg.coins}
    except Exception as e:  # noqa: BLE001
        # Satu kali gagal tidak dihitung error siklus; _feed_health yang memberi alarm kalau berlanjut.
        print(f"[cycle] predictedFundings gagal ({type(e).__name__}: {str(e)[:150]}); mark dan stop tetap jalan")
    if snap is not None:
        out["snapshot"] = ledger.record(snap, now)
    feed = _feed_health(ctx, snap, out)

    st = paper.load(cfg.paper["equity0"])
    st["started_utc"] = st.get("started_utc") or now.isoformat(timespec="seconds")
    cost, min_order = cfg.paper["cost_per_side"], cfg.paper["min_order_usd"]
    mids = None
    if st["positions"]:
        mids = info.all_mids()
        paper.mark(st, mids)
        for coin, pnl in paper.stops(st, mids, cfg.strategy, cost, now):
            ctx.outbox.add(f"🛑 <b>FCL paper — stop short {notify.esc(coin)}</b> (+{cfg.strategy['stop_pct']:.0%} dari entry), "
                           f"PnL {notify.usd(pnl)} USD")
        for coin in paper.close_missing(st, mids, cost, now):
            ctx.outbox.add(f"⚠️ <b>FCL paper — {notify.esc(coin)} hilang dari HYPE</b> (delisting?), "
                           f"posisi ditutup di harga terakhir")
        paper.funding(st, info, now)

    if feed.get("down_since") and _hours(now, feed["down_since"]) >= FEED_ALERT_H:
        _archive_fill(ctx, feed, feed["down_since"], now.isoformat())
    elif feed.get("backfill"):
        _archive_fill(ctx, feed, feed["backfill"]["since"], feed["backfill"]["until"])

    if rebalance_due(cfg, now, st.get("last_rebalance_day")):
        out["daily"] = _daily(ctx, st)

    hour = now.strftime("%Y-%m-%dT%H")
    if st.get("last_equity_hour") != hour:
        store.append("equity", {"time_utc": now.isoformat(timespec="seconds"), "event": "mark",
                                **paper.snapshot(st)})
        st["last_equity_hour"] = hour
    paper.save(st)
    return out


def _feed_health(ctx: Ctx, snap: dict | None, out: dict) -> dict:
    """Pantau feed BinPerp. state/feed.json: ok_utc, down_since, alert_utc, archive_utc, archive_days,
    backfill (celah yang masih diisi dari arsip setelah feed pulih)."""
    cfg, now = ctx.cfg, ctx.now
    fd = store.load_json(FEED, {}) or {}
    n = len(snap or {})
    if snap is not None and n >= FEED_MIN_SHARE * len(cfg.coins):
        if fd.get("down_since"):
            new = {"ok_utc": now.isoformat(timespec="seconds"),
                   **{k: fd[k] for k in ("archive_utc", "archive_days") if k in fd}}
            if _hours(now, fd["down_since"]) >= FEED_ALERT_H:
                new["backfill"] = {"since": fd["down_since"], "until": now.isoformat(timespec="seconds")}
            elif fd.get("backfill"):
                new["backfill"] = fd["backfill"]
            if fd.get("alert_utc"):
                ctx.outbox.add(f"✅ <b>FCL — funding Binance via HYPE masuk lagi</b> "
                               f"(putus sejak {notify.wib(fd['down_since'])}, {n} koin). "
                               f"Celahnya diisi dari arsip Binance begitu terbit.")
            fd = new
            store.save_json(FEED, fd)
        elif not fd.get("ok_utc") or _hours(now, fd["ok_utc"]) >= 1:
            fd["ok_utc"] = now.isoformat(timespec="seconds")
            store.save_json(FEED, fd)
        return fd
    if not fd.get("down_since"):
        fd["down_since"] = fd.get("ok_utc") or now.isoformat(timespec="seconds")
    down_h = _hours(now, fd["down_since"])
    if down_h >= FEED_ALERT_H:
        out["errors"].append(f"feed BinPerp putus {down_h:.0f} jam ({n} koin di snapshot)")
        if not fd.get("alert_utc") or _hours(now, fd["alert_utc"]) >= FEED_ALERT_EVERY_H:
            ctx.outbox.add(f"🚨 <b>FCL — funding Binance via HYPE tidak masuk</b> sejak {notify.wib(fd['down_since'])} "
                           f"({down_h:.0f} jam, {n}/{len(cfg.coins)} koin di snapshot)\n"
                           f"Sinyal F14 diisi dari arsip premium index Binance (data.binance.vision, tertinggal 1–2 hari, "
                           f"korelasi ranking F14 ±0,81, Sharpe forward ±1,1 vs 2,3 kalau putus > 14 hari). "
                           f"Cek <code>predictedFundings</code> HYPE.")
            fd["alert_utc"] = now.isoformat(timespec="seconds")
    store.save_json(FEED, fd)
    return fd


def _archive_fill(ctx: Ctx, fd: dict, since: str, until: str) -> int:
    """Cadangan feed: isi hari [since, until] dari arsip Binance, paling sering tiap ARCHIVE_EVERY_H jam."""
    cfg, now = ctx.cfg, ctx.now
    if fd.get("archive_utc") and _hours(now, fd["archive_utc"]) < ARCHIVE_EVERY_H:
        return 0
    fd["archive_utc"] = now.isoformat(timespec="seconds")
    store.save_json(FEED, fd)              # catat dulu: kalau siklus mati di tengah, tidak diulang tiap 10 menit
    done = set(fd.get("archive_days", []))
    first = max(dt.datetime.fromisoformat(since).date(), now.date() - dt.timedelta(days=cfg.strategy["funding_days"] + 1))
    last = dt.datetime.fromisoformat(until).date()
    want, d = [], first
    while d <= last:
        if d.isoformat() not in done:
            want.append(d)
        d += dt.timedelta(days=1)
    days = [d for d in want if d < now.date()][:ARCHIVE_MAX_DAYS]     # arsip hari ini belum terbit
    if not want and fd.pop("backfill", None):
        store.save_json(FEED, fd)
    if not days:
        return 0
    iv = bootstrap.known_intervals()
    iv = {c: iv.get(c, 8) for c in cfg.coins}
    try:
        n, complete = bootstrap.fill(cfg, days, iv, now, log=lambda *_: None)
    except Exception as e:  # noqa: BLE001
        print(f"[archive] gagal ({type(e).__name__})")
        return 0
    fd["archive_days"] = sorted(done | {d.isoformat() for d in complete})[-30:]
    if fd.get("backfill") and all(d.isoformat() in fd["archive_days"] for d in want):
        fd.pop("backfill")
    store.save_json(FEED, fd)
    print(f"[archive] {n} settlement dari arsip untuk {[d.isoformat() for d in days]} (lengkap: {len(complete)})")
    return n


def _daily(ctx: Ctx, st: dict) -> bool:
    cfg, info, now = ctx.cfg, ctx.info, ctx.now
    p = cfg.strategy
    day = now.date().isoformat()
    start = int((now - dt.timedelta(days=p["min_history_days"] + 20)).timestamp() * 1000)
    candles, failed = {}, []
    for coin in cfg.coins:
        try:
            candles[coin] = info.candles(coin, "1d", start)
        except Exception as e:  # noqa: BLE001
            failed.append(coin)
            print(f"[daily] candle {coin} gagal ({type(e).__name__})")
    h, m = cfg.rebalance_hm
    late_h = (now - now.replace(hour=h, minute=m, second=0, microsecond=0)).total_seconds() / 3600
    if len(failed) > CANDLE_FAIL_MAX * len(cfg.coins) and late_h < CANDLE_RETRY_H:
        if st.get("candle_alert_day") != day:
            ctx.outbox.add(f"⚠️ <b>FCL paper — rebalance ditunda</b>: candle {len(failed)}/{len(cfg.coins)} koin gagal diambil. "
                           f"Dicoba lagi tiap siklus, paling lama sampai {CANDLE_RETRY_H} jam setelah jadwal.")
            st["candle_alert_day"] = day
        print(f"[daily] ditunda: {len(failed)} candle gagal")
        return False
    f = ledger.f14(now, p["funding_days"])
    book = strategy.targets(candles, f, st["equity"], set(st["banned"]), p)
    mids = info.all_mids()
    paper.mark(st, mids)
    n = paper.rebalance(st, book.weights, mids, cfg.paper["cost_per_side"], cfg.paper["min_order_usd"], now)
    st["banned"] = []                     # larangan short setelah stop berlaku satu rebalance (sama dengan backtest)
    st["last_rebalance_day"] = day
    for row in book.table:
        store.append("targets", {"day": day, **row})
    snap = paper.snapshot(st)
    store.append("equity", {"time_utc": now.isoformat(timespec="seconds"), "event": "rebalance", **snap})
    pruned = ledger.prune(now)
    rows = ledger.rows_in_window(now, p["funding_days"])
    boot = sum(1 for r in rows if r["source"] == "bootstrap")
    longs = [r["coin"] for r in book.table if r["side"] == "long"]
    shorts = [r["coin"] for r in book.table if r["side"] == "short"]
    ret = snap["equity"] / st["equity0"] - 1
    warn = [book.note] if book.note else []
    if failed:
        warn.append(f"candle gagal {len(failed)} koin: {', '.join(failed[:10])}")
    ctx.outbox.add(
        f"📊 <b>FCL paper — rebalance {notify.wib(now)}</b>\n"
        f"Ekuitas <b>{notify.usd(snap['equity'])}</b> USD ({notify.pct(ret)} sejak mulai), DD {snap['dd_pct']:.1f}%\n"
        f"Universe {book.n_universe}, k = {book.k}/sisi, order {n}\n"
        f"🟢 Long: {notify.esc(', '.join(longs) or '-')}\n"
        f"🔴 Short: {notify.esc(', '.join(shorts) or '-')}\n"
        f"Fee kumulatif {notify.usd(snap['fees_cum'])}, funding {notify.usd(snap['funding_cum'])}\n"
        f"<i>Data funding 14 hari: {len(rows) - boot} rekaman bot, {boot} bootstrap arsip</i>"
        + "".join(f"\n⚠️ {notify.esc(w)}" for w in warn))
    print(f"[daily] universe={book.n_universe} k={book.k} orders={n} equity={snap['equity']:.2f} "
          f"pruned={pruned} candle_gagal={len(failed)}")
    return True
