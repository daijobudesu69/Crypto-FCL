# r2_candle_trades - round 2 (HYPE candle inputs only) - RESULTS

**Verdict: no edge in either family. Nothing to forward.** Every config that looks positive turns out
to be momentum (corr with 14d XS momentum 0.4-0.6, alpha t-stat vs xsmom14+BTC under 2), a size bet,
or a spike that falls apart in 2023-24. Once the signals are residualised against own-horizon and 14d
returns, both families are flat to negative.

Setup (applies to every config): IS 2020-01-01..2024-12-31 via lab.py only. Signal inputs are only
open/high/low/close/qv/trades. Universe `lab.hype_universe("1d", top_k=50)`, flat on days with <10
names, so trading starts 2020-04. Quintile L/S, equal weight, gross 1.0, daily rebalance, cost
`lab.HYPE_COST`, funding charged as a cost. That gives about 14 positions on average, so the minimum
equity is about $140. Configs run: A 42, B 34, plateau/diagnostic extras 3, **total 79**. The deflated
Sharpe (DSR) uses n=78.

Scripts: `r2common.py` (helpers, NW alpha regression), `famA_grid.py`, `famB_grid.py`, `checks.py`
(stress tests, 8L/8S, top30/75, venue check), `diag_atslvl.py`. Logs: `famA_grid.csv`, `famB_grid.csv`,
`checks.csv`, `checks.json`.

## Family A - candle shape / accumulation (42 configs)
Hypothesis: closes that sit persistently near the high (CLV), small upper wicks or large lower wicks
mean patient buyers are absorbing supply (continuation), or late buyers are exhausted (reversal).
Grid: {CLV, UW, LW, WI=LW-UW, RPOS} x N{5,10,20} x direction, plus CLV/WI residualised on rank(ret_N),
rank(ret_14).

- The CLV continuation results are mostly momentum. CLV20+ has SR 0.89, but corr to xsmom is 0.41,
  alpha t is 1.17, 2023-24 SR is -0.42, lag-1 SR is 0.35 and 8L/8S SR is 0.57.
  **The residualised CLV/WI configs are all <= 0.10.** The reversal directions are all strongly negative.
- RPOS20+ (range position, i.e. stochastic %K) has SR 1.44, but it is momentum: corr 0.63, a
  vol-normalised 20d relative strength close to what RNT already does. Its 2023-24 SR is 0.05, so it
  is not new and not forwardable.
- LW5- (short coins with long lower wicks) has SR 0.84, but 2022 is -0.73, 2x-cost SR is 0.20, the
  neighbours sit at 0.18 (a spike), and all of it comes from the short leg.
- Upper wicks: nothing (|SR| < 0.2 in the hypothesis direction).
- Venue translation is fine for this family. The XS rank correlation of the features between Binance
  and HYPE candles in 2024 is 0.98-0.99, so the problem is no edge, not data.

## Family B - participation / average trade size (34 + 2 configs)
Hypothesis: a surge in trade count with a falling average trade size means retail FOMO, so those coins
underperform. A rising average size means larger players are accumulating, so those coins outperform.
FOMO_h = TCchg_h - ATSchg_h is orthogonal to the plain volume shock that round 1 ruled out.

- **The retail-FOMO hypothesis is rejected.** Shorting high FOMO loses money at every horizon
  (SR -0.37/-0.42/-0.40). Residualised, it is about 0 (0.04/0.19/-0.16).
- ATSchg7+ (long rising USD average trade size) has SR 0.84 and is positive in every sub-period
  (1.20/0.80/0.39). It holds up at 2x cost (0.62) and lag 1 (0.76), but **it is momentum**: corr 0.51,
  alpha t vs mom+BTC only 0.32 (alpha +3.6%/yr). USD trade size rises mechanically with price (XS rank
  corr with ret_7 is 0.24). Residualised on returns it goes negative (-0.54/-0.34/-0.45). It is
  worthless to someone who already runs RMF/RNT.
- ATSlvl- (long small average-trade-size coins, short large) has SR 0.69, low turnover, 2x-cost SR 0.63,
  lag-1 SR 0.59 and 8L/8S SR 0.88. But it is a **size/liquidity bet**: XS rank corr with log qv is 0.88,
  and it is short BTC/ETH/XRP/SOL/BNB almost every day. 2023-24 is -0.46 (2024 -22%), the long leg vs
  the equal-weight universe is -0.21, and it is fully exposed to survivorship bias, because the long
  side is small coins that we know survived to 2026.
- Translation risk: live trade counts would come from HYPE candles (`n`), and HYPE fill microstructure
  differs from Binance's. Trades are not in the hype_1d cache, so rank stability could not be checked.

## Best attempt per family (all stress numbers on hype top-50)
| spec | SR | 20-21/22/23-24 | 2x | lag1 | 8L8S | top30/75 | cMOM | alpha t (mom+BTC) | DSR | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| A CLV20+ | 0.89 | 1.91/0.55/-0.42 | 0.59 | 0.35 | 0.57 | 1.11/0.88 | 0.41 | 1.17 | 0.23 | none (momentum + dead 23-24) |
| A RPOS20+ (ref) | 1.44 | 2.73/0.73/0.05 | 0.93 | 0.66 | 0.96 | 1.43/1.51 | 0.63 | 2.05 | 0.69 | none (= vol-scaled momentum) |
| B ATSchg7+ | 0.84 | 1.20/0.80/0.39 | 0.62 | 0.76 | 0.69 | 0.81/0.89 | 0.51 | 0.32 | 0.20 | weak (momentum) |
| B ATSlvl- | 0.69 | 1.48/0.54/-0.46 | 0.63 | 0.59 | 0.88 | 0.80/0.82 | 0.08 | 1.11 | 0.10 | weak (size + survivorship) |

One pattern repeats across every family: 2023 is the worst year for all of the XS L/S books.
