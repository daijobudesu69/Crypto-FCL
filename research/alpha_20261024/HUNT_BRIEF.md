# Alpha hunt brief (2026-10-24) — read fully before doing anything

You are one quant researcher on a desk hunting for ONE tradable crypto alpha. Act like a
professional: economic rationale first, small pre-registered grids, honest negative results.

## Hard rules

1. **Research period = 2020-01-01 → 2024-12-31 only.** Load data ONLY through `lab.py` (it
   truncates at 2025-01-01). Never set `ALPHA_FORWARD`, never read `..\data\*.parquet` or other
   folders' result files that contain 2025+ performance. The 2025+ forward test is run once by
   the lead on the frozen spec. Peeking ruins the whole exercise.
2. **Pre-register.** Write your grid down in your script header BEFORE running it. Max ~40
   configs per hypothesis family. Count every config you ran (needed for the deflated Sharpe).
3. **No lookahead.** Panel row t = known at close of bar t. `lab.backtest(W)` trades W[t] at
   close t and earns bar t+1. Anything that uses future info (full-sample z-scores, full-sample
   ranks, rolling windows with center=True, universe chosen by future volume) is a bug.
   Rolling stats must be trailing only.
4. **Costs are real.** Default 0.07%/side on drift-adjusted turnover + actual Binance funding
   paid/received. Also report Sharpe at 2× cost (0.14%/side) and with `extra_lag=1`.
5. **Memory is tight (≈1–2 GB free on a 16 GB box shared by 5 researchers).** Use 1d or 4h
   panels where possible. Load only the fields you need. 1h panels are ~36 MB/field — OK for a
   few fields, not for 15. Run one Python process at a time. No multiprocessing.
6. Work only inside your own folder `alpha_20261024/hunt/<your_id>/`. Do not edit `lab.py`,
   `build_cache.py` or the cache; if you need a helper, write it in your folder.
7. Python: `python` (3.14, pandas 3, numpy 2.4, scipy, numba available; no statsmodels).

## lab.py API (alpha_20261024/lab.py)

```python
import sys; sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab
P = lab.load(field, tf)    # tf in "1d","4h","1h"; wide float32 DataFrame, index = bar OPEN ts UTC
# fields: open high low close qv(quote vol USD) tbqv(taker-buy quote vol) trades
#         fund (funding settled during bar, sum, fraction e.g. 0.0001 = 0.01%)
#         oi_usd top_pos_ls top_acct_ls global_ls taker_ls_vol  (Binance metrics, as-of bar close;
#                 BTC/ETH from 2020-09, most alts from 2021-12 -> metrics factors are 2022–2024 only)
#         prem (perp premium index, last hourly close in bar)  prem_mean (mean of hourly closes in bar)
#         spot_close spot_qv  (1d only, Binance spot)
U = lab.universe("1d", min_hist_days=60, top_k=50)      # point-in-time liquidity universe (bool)
lab.quantile_ls_weights(score, U, q=0.2)               # long top quintile, short bottom, $-neutral
lab.xs_demean_weights(score, U)                        # rank/score-proportional $-neutral
res = lab.backtest(W, "1d", cost=0.0007, extra_lag=0, rebalance_every=1)
res["metrics"]  # cagr vol sharpe sortino maxdd calmar t_nw skew kurt pct_pos_months turnover_per_year
                # cost_drag_per_year funding_per_year avg_gross avg_net avg_n_pos
res["df"]       # per-bar r, gross_r, fund_r, cost_r, turnover, gross_exposure, net_exposure, n_pos
lab.yearly(res["df"]["r"], "1d"); lab.report(res, "name")
lab.deflated_sharpe(sharpe_ann, n_trials, T_days, skew, kurt_raw)   # kurt_raw = excess + 3
```

Universe = 150 Binance USDT perps still alive in Sep 2026, no stablecoins (survivorship bias,
accepted by the user). It inflates long-side results and understates short-side results. Say
how your idea is exposed to it.

## What already exists in this folder (do not re-discover; use as controls only)

- Single price indicators (EMA/RSI/BB/VWAP/SMC…) have no edge after costs on 15m/1h.
- CLR: 4h liquidation-flush basket long (≥10 coins re-enter lower BB, vol spike, OI −5%/24h) and
  4h trapped-long breakdown short (Donchian-20 break, EMA50<EMA200, funding > 0.01%).
- RMF: 14-day cross-sectional momentum rotation long-only (top 10, BTC>EMA50 filter).
- RNT: risk-adjusted relative-strength long/short on top-20 + long-only channel trend on top-5.
- MEX: 4h momentum-exhaustion breakout on 13 coins.
- Stat-arb pairs (BTC vs alts, alt vs alt cointegration): no edge.
So plain cross-sectional momentum and channel trend are NOT new. If your factor's returns
correlate > 0.5 with 14d XS momentum, say so.

## What to deliver (your final answer = structured output)

For each family you were assigned: the hypothesis + rationale (who is on the other side and why
it persists), the grid, results, and your best candidate (or "none"). For each candidate give:
exact rules (signal, universe, weighting, rebalance, exits), IS metrics (2020–2024 or the
available sub-range), Sharpe by sub-period (2020–21, 2022, 2023–24 where available), Sharpe at 2×
cost and at extra_lag=1, median Sharpe of the neighbouring parameter configs (plateau check),
long-leg vs short-leg Sharpe, daily return correlation with BTC and with 14d XS momentum (top-50
quintile L/S), avg number of positions, minimum equity needed if each position must be ≥ $10
(Hyperliquid minimum order), number of configs tried, deflated Sharpe probability, and your
honest verdict + concerns. Save your scripts and a short `RESULTS.md` in your folder.

A candidate worth forwarding: IS Sharpe ≥ ~1.0 net, positive in every sub-period, survives 2×
cost and lag 1 with Sharpe ≥ ~0.6, plateau not a spike, and a rationale that makes sense. Most
ideas will fail — report failures plainly; a clean "no edge" is a valuable result.

## UPDATE (user, mid-run): execution venue = Hyperliquid DEX ("HYPE")

- Only 92 of the 150 Binance coins trade on HYPE (37 of the Binance top-50). Use
  `lab.hype_universe("1d", top_k=...)` for the tradable version of any candidate and report its
  metrics alongside the full-universe result. `lab.HYPE_COST` = 0.065%/side (taker 0.045% + slip).
- Minimum order on HYPE is $10, starting equity $200 -> at gross 1.0 at most ~20 positions,
  realistically 6–12. Prefer designs that work with few names.
- Live signal data must be available on HYPE: candles, volume, funding (hourly) are; Binance
  long/short ratios and OI history are NOT reachable live from the user's network (only via the
  data.binance.vision archive with ~1 day delay). A metrics-based signal is therefore a
  deployment risk — flag it.

## ROUND 2 (user constraint, 2026-10-24): HYPE candles only, no VPS

The user will NOT run a VPS and Binance is geoblocked. A deployable strategy must compute its signal in a
daily (or 4h) GitHub Actions job from **HYPE candle data only: open, high, low, close, volume (qv),
trades count** — fields `open high low close qv trades` in lab. **Forbidden as signal inputs:** fund,
oi_usd, top_pos_ls, top_acct_ls, global_ls, taker_ls_vol, prem, prem_mean, spot_close, spot_qv, tbqv
(HYPE candles have no taker-buy volume). Funding is still charged as a COST by lab.backtest (keep it).
Backtests use the Binance OHLCV history (HYPE prices match within ~5 bp). Universe:
`lab.hype_universe("1d", top_k=...)`, cost `lab.HYPE_COST`. Exchange-resting stop orders are allowed.
Signals that need sub-hour monitoring are not (cron can run 5–30 min late: check `extra_lag`).

Round-1 results already rule out (do not redo): 1–5d and 4h short-term reversal (continuation instead,
reversal only harvestable as a maker), 1–5d continuation (= momentum), BAB/IVOL/MAX/skew textbook
directions, volume-shock, hour-of-day / weekday timing, BTC->alt lead-lag, vol-managed basket,
plain XS momentum (RMF/RNT), channel trend (RNT), Donchian breakout (MEX/CLR). The positive skew
flip (long +skew / short −skew 4h returns, Sharpe 0.86) was weak.
Breadth matters: round 1 showed concentrated books (4–6 names/side) lose most of the edge. With $200
and a $10 minimum, ~16–20 positions is the practical maximum at gross 1.0–1.2.
