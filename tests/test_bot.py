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
