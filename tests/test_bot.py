import datetime as dt

from fclbot import config, control, jobs, ledger, notify, paper, store, strategy
from tests.fakes import FakeInfo, candles

P = config.load().strategy
UTC = dt.timezone.utc


def test_ledger_settles_snapshot_and_prefers_predicted():
    t0 = dt.datetime(2026, 10, 24, 7, 55, tzinfo=UTC)
    settle = int(dt.datetime(2026, 10, 24, 8, tzinfo=UTC).timestamp() * 1000)
    store.append("funding_ledger", {"settle_utc": ledger.iso(settle), "settle_ms": settle, "coin": "SOL", "rate": 0.5,
                                    "interval_h": 8, "source": "bootstrap", "seen_utc": ""})
    ledger.record({"SOL": {"rate": 0.0001, "next_ms": settle, "interval_h": 8}}, t0)
    assert store.read("funding_ledger")[-1]["source"] == "bootstrap"           # belum settle
    ledger.record({"SOL": {"rate": 0.0002, "next_ms": settle + 8 * 3_600_000, "interval_h": 8}},
                  t0 + dt.timedelta(minutes=10))
    rows = ledger.rows_in_window(t0 + dt.timedelta(minutes=10), 14)
    assert len(rows) == 1 and rows[0]["source"] == "predicted" and float(rows[0]["rate"]) == 0.0001
    sig, cov = ledger.f14(t0 + dt.timedelta(minutes=10))["SOL"]
    assert abs(sig - 0.0001 / 8 * 336) < 1e-12 and abs(cov - 8 / 336) < 1e-12


def _book(n=30, equity=200.0, banned=()):
    coins = [f"C{i:02d}" for i in range(n)]
    cs = {c: candles(seed=i, vol=1e6 * (n - i)) for i, c in enumerate(coins)}
    f14 = {c: (i * 0.001, 1.0) for i, c in enumerate(coins)}               # C29 = funding tertinggi
    return coins, cs, strategy.targets(cs, f14, equity, set(banned), P)


def test_targets_shape_and_neutrality():
    coins, cs, b = _book()
    assert b.k == 8 and b.n_universe == 30
    longs = [c for c, w in b.weights.items() if w > 0]
    shorts = [c for c, w in b.weights.items() if w < 0]
    assert len(longs) == 8 and len(shorts) == 8
    assert abs(sum(b.weights.values())) < 1e-12 and abs(sum(abs(w) for w in b.weights.values()) - 1.0) < 1e-12
    assert set(longs) == {f"C{i:02d}" for i in range(8)}                    # funding terendah


def test_momentum_filter_and_ban_exclude_shorts():
    coins, cs, b = _book()
    u = strategy.universe(cs, {c: (i * 0.001, 1.0) for i, c in enumerate(coins)}, P)
    u["r"] = u["mom14"].rank(pct=True)
    hot = set(u.loc[u["r"] > P["mom_skip_pct"], "coin"])
    shorts = {c for c, w in b.weights.items() if w < 0}
    assert hot and not (hot & shorts)
    top = sorted(shorts)[-1]
    _, _, b2 = _book(banned=[top])
    assert top not in {c for c, w in b2.weights.items() if w < 0}


def test_k_scales_with_equity_and_flat_when_small_universe():
    assert _book(equity=1000)[2].k == 10                                     # N//3 = 10 membatasi
    assert _book(n=10)[2].weights == {}


def test_paper_rebalance_mark_stop_and_funding():
    now = dt.datetime(2026, 10, 24, 8, 20, tzinfo=UTC)
    st = paper.load(200.0)
    n = paper.rebalance(st, {"A": 0.25, "B": -0.25}, {"A": 10.0, "B": 20.0}, 0.00065, 10.0, now)
    assert n == 2 and abs(st["equity"] - (200 - 100 * 0.00065)) < 1e-9
    paper.mark(st, {"A": 11.0, "B": 20.0})
    assert abs(st["price_pnl_cum"] - 5.0) < 1e-9                              # 5 koin A x +1
    hit = paper.stops(st, {"A": 11.0, "B": 28.5}, P, 0.00065, now)
    assert hit and "B" not in st["positions"] and st["banned"] == ["B"]
    t = int((now + dt.timedelta(hours=1)).timestamp() * 1000)
    info = FakeInfo(["A"], fh={"A": [(t, 0.001)]})
    paid = paper.funding(st, info, now + dt.timedelta(hours=1, minutes=5))
    assert paid < 0                                                         # long bayar funding positif


def test_small_adjustments_are_skipped():
    now = dt.datetime(2026, 10, 24, 8, 20, tzinfo=UTC)
    st = paper.load(200.0)
    paper.rebalance(st, {"A": 0.25}, {"A": 10.0}, 0.0, 10.0, now)
    assert paper.rebalance(st, {"A": 0.27}, {"A": 10.0}, 0.0, 10.0, now) == 0   # selisih $4 < $10


def test_cycle_end_to_end_rebalances_once_per_day():
    cfg = config.load()
    coins = sorted(cfg.coins)[:30]
    cfg.coins = {c: cfg.coins[c] for c in coins}
    now = dt.datetime(2026, 10, 24, 8, 25, tzinfo=UTC)
    for i, c in enumerate(coins):
        for d in range(42):
            t = int((now - dt.timedelta(hours=8 * d + 1)).timestamp() * 1000)
            store.append("funding_ledger", {"settle_utc": ledger.iso(t), "settle_ms": t, "coin": c, "rate": i * 1e-5,
                                            "interval_h": 8, "source": "bootstrap", "seen_utc": ""})
    info = FakeInfo(coins, pf={c: {"rate": 1e-4, "next_ms": int(now.timestamp() * 1000) + 3_600_000, "interval_h": 8}
                               for c in coins})
    ctx = jobs.Ctx(cfg=cfg, info=info, ctrl=control.Control(), outbox=notify.Outbox(now), now=now)
    assert jobs.run(ctx)["daily"] is True
    st = store.load_json("paper.json")
    assert len(st["positions"]) == 16 and st["last_rebalance_day"] == "2026-10-24"
    ctx.now = now + dt.timedelta(minutes=10)
    assert jobs.run(ctx)["daily"] is False


def test_rebalance_due_time():
    cfg = config.load()
    d = dt.datetime(2026, 10, 24, tzinfo=UTC)
    assert not jobs.rebalance_due(cfg, d.replace(hour=8, minute=19), None)
    assert jobs.rebalance_due(cfg, d.replace(hour=8, minute=20), None)
    assert not jobs.rebalance_due(cfg, d.replace(hour=9), "2026-10-24")


def test_control_reserved_modes_fall_back_to_paper(tmp_path):
    p = tmp_path / "bot.yaml"
    p.write_text('mode: "live"\n')
    c = control.read((str(p),))
    assert c.mode == "paper" and c.problem


# ---------------------------------------------------------------- infra: feed funding, delisting, candle
def _cycle_setup(now, n=30):
    cfg = config.load()
    coins = sorted(cfg.coins)[:n]
    cfg.coins = {c: cfg.coins[c] for c in coins}
    for i, c in enumerate(coins):
        for d in range(42):
            t = int((now - dt.timedelta(hours=8 * d + 1)).timestamp() * 1000)
            store.append("funding_ledger", {"settle_utc": ledger.iso(t), "settle_ms": t, "coin": c, "rate": i * 1e-5,
                                            "interval_h": 8, "source": "bootstrap", "seen_utc": ""})
    pf = {c: {"rate": 1e-4, "next_ms": int(now.timestamp() * 1000) + 3_600_000, "interval_h": 8} for c in coins}
    return cfg, coins, pf


def _ctx(cfg, info, now):
    return jobs.Ctx(cfg=cfg, info=info, ctrl=control.Control(), outbox=notify.Outbox(now), now=now)


def _texts(ctx):
    return " ".join(it["text"] for it in ctx.outbox.items)


def test_funding_failure_does_not_block_mark_and_stops():
    now = dt.datetime(2026, 10, 24, 10, 0, tzinfo=UTC)
    cfg = config.load()
    st = paper.load(200.0)
    paper.rebalance(st, {"A": 0.25, "B": -0.25}, {"A": 10.0, "B": 20.0}, 0.00065, 10.0, now)
    st["last_rebalance_day"] = "2026-10-24"
    paper.save(st)
    info = FakeInfo(["A", "B"], mids={"A": 10.0, "B": 28.5}, pf=RuntimeError("HYPE /info predictedFundings gagal"))
    ctx = _ctx(cfg, info, now)
    out = jobs.run(ctx)                                                     # tidak raise
    st = store.load_json("paper.json")
    assert "B" not in st["positions"] and st["banned"] == ["B"]             # stop tetap jalan
    assert out["errors"] == [] and out["snapshot"] is None                   # satu kali gagal belum error
    assert store.load_json("feed.json")["down_since"]


def test_feed_down_alerts_fills_from_archive_and_backfills_after_recovery(monkeypatch):
    from fclbot import bootstrap
    calls = []

    def fake_premium(base, sym, day, session=None):
        calls.append((sym, day))
        t0 = int(dt.datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp() * 1000)
        return {t0 + h * 3_600_000: 0.0002 for h in range(24)}

    monkeypatch.setattr(bootstrap, "premium", fake_premium)
    now = dt.datetime(2026, 10, 24, 6, 0, tzinfo=UTC)
    cfg, coins, pf = _cycle_setup(now)
    store.save_json("feed.json", {"ok_utc": "2026-10-23T20:00:00+00:00"})
    info = FakeInfo(coins, pf={})                                            # respons ada, BinPerp hilang
    ctx = _ctx(cfg, info, now)
    out = jobs.run(ctx)
    assert "tidak masuk" in _texts(ctx) and any("feed BinPerp" in e for e in out["errors"])
    fd = store.load_json("feed.json")
    assert fd["archive_days"] == ["2026-10-23"]                              # hari ini belum terbit
    new = [r for r in store.read("funding_ledger") if r["seen_utc"] == now.isoformat(timespec="seconds")]
    assert new and all(r["source"] == "bootstrap" and r["settle_utc"] >= "2026-10-23" for r in new)
    n_calls = len(calls)
    jobs.run(_ctx(cfg, info, now + dt.timedelta(minutes=10)))
    assert len(calls) == n_calls                                             # maks tiap 6 jam

    rec = now + dt.timedelta(minutes=30)
    info.pf = pf
    ctx = _ctx(cfg, info, rec)
    assert jobs.run(ctx)["errors"] == []
    assert "masuk lagi" in _texts(ctx)
    fd = store.load_json("feed.json")
    assert "down_since" not in fd and fd["backfill"]["since"] == "2026-10-23T20:00:00+00:00"

    nxt = dt.datetime(2026, 10, 25, 1, 0, tzinfo=UTC)                         # arsip 24 Okt sudah terbit
    jobs.run(_ctx(cfg, info, nxt))
    fd = store.load_json("feed.json")
    assert "2026-10-24" in fd["archive_days"] and "backfill" not in fd
    assert {d for _, d in calls[n_calls:]} == {dt.date(2026, 10, 24)}


def test_stale_candles_are_excluded_from_universe():
    coins = [f"C{i:02d}" for i in range(30)]
    cs = {c: candles(seed=i, vol=1e6 * (30 - i)) for i, c in enumerate(coins)}
    cs["C00"] = candles(seed=0, vol=1e9, end="2026-10-20")                  # berhenti trading 4 hari lalu
    u = strategy.universe(cs, {c: (i * 0.001, 1.0) for i, c in enumerate(coins)}, P)
    assert "C00" not in set(u["coin"]) and len(u) == 29


def test_position_missing_from_mids_is_closed_after_three_cycles():
    now = dt.datetime(2026, 10, 24, 8, 20, tzinfo=UTC)
    st = paper.load(200.0)
    paper.rebalance(st, {"X": 0.25, "Y": -0.25}, {"X": 10.0, "Y": 20.0}, 0.0, 10.0, now)
    mids = {f"M{i}": 1.0 for i in range(60)} | {"Y": 20.0}
    assert paper.close_missing(st, {"Y": 20.0}, 0.0, now) == []                # allMids tidak lengkap: abaikan
    assert paper.close_missing(st, mids, 0.0, now) == [] and paper.close_missing(st, mids, 0.0, now) == []
    assert paper.close_missing(st, mids, 0.0, now) == ["X"]
    assert set(st["positions"]) == {"Y"} and store.read("orders")[-1]["kind"] == "close"


def test_rebalance_postponed_when_many_candles_fail():
    now = dt.datetime(2026, 10, 24, 8, 25, tzinfo=UTC)
    cfg, coins, pf = _cycle_setup(now)
    info = FakeInfo(coins, pf=pf, fail=coins[:4])                            # 4/30 > 10%
    ctx = _ctx(cfg, info, now)
    assert jobs.run(ctx)["daily"] is False and "ditunda" in _texts(ctx)
    assert store.load_json("paper.json")["positions"] == {}
    ctx = _ctx(cfg, info, now.replace(hour=12, minute=30))                   # > 4 jam: jalan dengan data yang ada
    assert jobs.run(ctx)["daily"] is True
    assert len(store.load_json("paper.json")["positions"]) == 16 and "candle gagal 4" in _texts(ctx)


def test_notify_without_telegram_writes_actions_annotation(monkeypatch, capsys):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert notify.send_now("🚨 <b>FCL</b> 50% putus\nbaris dua")
    line = [x for x in capsys.readouterr().out.splitlines() if x.startswith("::notice")][0]
    assert line.endswith("::🚨 FCL 50%25 putus%0Abaris dua")
