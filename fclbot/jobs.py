"""Satu siklus FCL: rekam funding -> mark buku paper -> stop -> funding HYPE -> rebalance harian."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from . import ledger, notify, paper, store, strategy


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


def run(ctx: Ctx) -> dict:
    cfg, info, now = ctx.cfg, ctx.info, ctx.now
    out = {"snapshot": None, "daily": False}
    if ctx.ctrl.mode == "off":
        return out

    snap = {c: v for c, v in info.predicted_fundings().items() if c in cfg.coins}
    out["snapshot"] = ledger.record(snap, now)

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
        paper.funding(st, info, now)

    if rebalance_due(cfg, now, st.get("last_rebalance_day")):
        _daily(ctx, st)
        out["daily"] = True

    hour = now.strftime("%Y-%m-%dT%H")
    if st.get("last_equity_hour") != hour:
        store.append("equity", {"time_utc": now.isoformat(timespec="seconds"), "event": "mark",
                                **paper.snapshot(st)})
        st["last_equity_hour"] = hour
    paper.save(st)
    return out


def _daily(ctx: Ctx, st: dict) -> None:
    cfg, info, now = ctx.cfg, ctx.info, ctx.now
    p = cfg.strategy
    start = int((now - dt.timedelta(days=p["min_history_days"] + 20)).timestamp() * 1000)
    candles = {}
    for coin in cfg.coins:
        try:
            candles[coin] = info.candles(coin, "1d", start)
        except Exception as e:  # noqa: BLE001
            print(f"[daily] candle {coin} gagal ({type(e).__name__})")
    f = ledger.f14(now, p["funding_days"])
    book = strategy.targets(candles, f, st["equity"], set(st["banned"]), p)
    mids = info.all_mids()
    paper.mark(st, mids)
    n = paper.rebalance(st, book.weights, mids, cfg.paper["cost_per_side"], cfg.paper["min_order_usd"], now)
    st["banned"] = []                     # larangan short setelah stop berlaku satu rebalance (sama dengan backtest)
    day = now.date().isoformat()
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
    ctx.outbox.add(
        f"📊 <b>FCL paper — rebalance {notify.wib(now)}</b>\n"
        f"Ekuitas <b>{notify.usd(snap['equity'])}</b> USD ({notify.pct(ret)} sejak mulai), DD {snap['dd_pct']:.1f}%\n"
        f"Universe {book.n_universe}, k = {book.k}/sisi, order {n}\n"
        f"🟢 Long: {notify.esc(', '.join(longs) or '-')}\n"
        f"🔴 Short: {notify.esc(', '.join(shorts) or '-')}\n"
        f"Fee kumulatif {notify.usd(snap['fees_cum'])}, funding {notify.usd(snap['funding_cum'])}\n"
        f"<i>Data funding 14 hari: {len(rows) - boot} rekaman bot, {boot} bootstrap arsip</i>"
        + (f"\n⚠️ {notify.esc(book.note)}" if book.note else ""))
    print(f"[daily] universe={book.n_universe} k={book.k} orders={n} equity={snap['equity']:.2f} pruned={pruned}")
