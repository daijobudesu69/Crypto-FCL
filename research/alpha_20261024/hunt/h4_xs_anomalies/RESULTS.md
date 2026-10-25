# h4_xs_anomalies: short-term reversal and low-risk/lottery anomalies (IS 2020-2024)

**Bottom line: none of these is worth forwarding.** No config meets the bar: net Sharpe of at least ~1.0, positive in every sub-period, at least ~0.6 at 2x cost and at lag 1, a plateau, and a rationale that holds up.
Two configs are marked *weak* and are kept for the lead to see: BAB beta-neutral, and the sign-flipped "lottery premium". Each one fails a sub-period. Each one is mostly explained by known exposures or by a handful of surviving coins. Neither has a deflated Sharpe probability above 0.21.

Data and protocol: all data was loaded through `lab.py`, so it stops at 2025-01-01. `ALPHA_FORWARD` was never set and no forward data was used.
One process deviation: early on I ran `pd.read_parquet` directly on three `cache/4h` files to check that the 4h build had finished. I printed only shape, dtype and the first and last index timestamps. No values were looked at or used. Every later load went through `lab`.

Universe: `lab.universe(tf, min_hist_days=60, top_k=50)`. Family B excludes BTC, because BTC is the factor (its beta is 1 and its idiosyncratic vol is 0 by construction). Costs are 0.07% per side plus real funding. Stress tests are 2x cost (0.14%) and `extra_lag=1`.

Scripts and logs in this folder:
- `famA_reversal.py`, `famA_flip.py`, `famB_lowrisk.py`, `famB_rev.py`, `famB_shortleg.py`: the grids. Each script's header pre-registers its grid.
- `summarize.py`, `diag_*.py`, `chk_flip.py`: diagnostics.
- `grid_log.csv`: every config run (94).
- `grid_table.md`: a compact table of all 94.

**Trial count.**
- Configs actually run: 94. Family A: 40 + 4. Family B: 36 + 10 + 3 robustness + 1 post-hoc.
- For deflated Sharpe I count trials as A = 84 and B = 86, or 170 in total. The flipped signs were chosen *after* seeing the pre-registered grids, so each original grid is counted twice.

---

## Family A: short-term reversal (liquidity provision)

**Hypothesis.** Liquidity-demanding flow (liquidations, retail market orders) pushes prices away from value. Whoever supplies liquidity earns the reversion.
Moves on low abnormal volume should revert, because they carry no information. Moves on high abnormal volume should continue (Campbell-Grossman-Wang; Llorente et al.). Residual returns versus BTC beta should isolate the idiosyncratic, liquidity-driven part.

**Grid: 40 configs, all pre-registered.**

| Group | Bars | Formation window | Return type | Execution / split | Configs |
|---|---|---|---|---|---|
| A1 | 1d | L = 1, 2, 3, 5 days | RAW / RESID | fresh daily book, or 3-day overlapping books | 16 |
| A2 | 1d | L = 1, 3 days | RAW / RESID | low vs high abnormal-volume half (q = 0.3 inside the half) | 8 |
| A3 | 4h | 4h, 12h, 24h | RAW / RESID | every bar, or 6-bar overlapping books | 12 |
| A4 | 4h | 12h, 24h | RAW | low vs high abnormal-volume half | 4 |

**Results.**
- **1d: there is no reversal anywhere. The cross-section shows continuation.** Gross Sharpe of the reversal book is -0.4 to -1.4 at every formation window. Net Sharpe is -1.2 to -2.2.
  - Volume conditioning behaves as the hypothesis says for high-volume moves, which continue strongly (A2 L3 HIGHAV: reversal gross -1.35). Low-volume moves do not revert either (gross -0.1 to -0.4).
- **4h: the liquidity-provision premium exists before costs and is far too small to trade.**
  - A3 h1 RAW (reversal over one 4h bar): gross +31%/yr, Sharpe +1.01. But turnover is about 3,400x per year, so cost drag is **236%/yr** and net Sharpe is -6.8.
  - A4 h3 LOWAV (12h moves on low abnormal volume): gross +29%/yr, Sharpe +1.14, against cost drag of **157%/yr**. The same moves on high abnormal volume show continuation (gross Sharpe -1.32), which matches the hypothesis.
  - Break-even cost is about **0.9 bp per side**.
  - The signal is gone within one bar (lag-1 Sharpe is worse still). Six-bar overlapping books cut cost to about 39%/yr, but they also turn gross Sharpe negative.
- **Flipped sign, A-flip (4 configs): short-horizon, volume-confirmed momentum.**
  - Gross Sharpe is about 1.0, net 0.30 to 0.51.
  - 2022 is badly negative (about -2.0).
  - At 2x cost the Sharpe is at most 0.0.
  - Correlation with 14d XS momentum is 0.38 to 0.57.
  - Spanning regression on BTC plus xsmom14: alpha is **-2% to -6% per year** (t from -0.2 to -0.5), with loading 0.45 on xsmom14. **It is the existing momentum factor, not new.**
- **Verdict: none.** The only part of family A that fits the reversal story is the gross 4h premium in low-volume moves. Only a sub-bar maker with rebates could harvest it, not a taker with $200.

## Family B: low-risk and lottery anomalies

**Hypothesis.** Leverage-constrained, lottery-seeking retail traders overpay for high-beta, high-IVOL, high-MAX and positively skewed coins. The trade is to go long the boring coins and short the lottery coins.

**Grid: 36 configs, all pre-registered.**
- Signals:
  - BAB: beta over 60 or 90 days.
  - IVOL: residual vol vs BTC over 30 or 60 days.
  - MAX: MAX1 (the largest daily return in 30 days) or MAX5 (the mean of the 5 largest daily returns in 30 days).
  - SKEW: skewness of daily returns over 30 or 90 days, or of 4h returns over 30 days.
- Construction: $-neutral quintiles (DN), or beta-neutral legs in the Frazzini-Pedersen style (BN). BN uses shrunk 60d beta, each leg scaled to beta 0.5, leg gross capped at 1.
- Rebalancing: daily (R1) or weekly (R7).

**Results.**
- **The textbook direction is rejected in all 28 IVOL/MAX/SKEW configs.** Net Sharpe ranges from -1.32 to 0.00.
- Over 2020-24, lottery-like coins *out*performed. The only year where boring coins won was the 2022 bear market (IVOL DN: +1.4 to +2.0 in 2022, negative in every other period).
- The BAB DN configs are also all negative.
- **Only BAB-BN is positive.** The best is `B_BAB_b90_BN_R7`, with net Sharpe 0.83, 0.77 at 2x cost and 0.80 at lag 1.
  - Neighbouring configs: 0.76, 0.55 and -0.08, median 0.55.
  - **2023-24 Sharpe is -0.70** (2024: -28%).
  - Yearly returns: 2020 +46%, 2021 +173%, 2022 +14%, 2023 -4%, 2024 -28%.
- **Why BAB-BN is not trusted.**
  - The beta-sorted legs earn about 0 versus the EW universe on their own.
  - The **top 5 coins provide 125% of all price PnL**: SOL, AVAX, EGLD, AAVE and FIL, held long in 2021.
  - In crypto, "low beta" partly means "low correlation". Newly listed narrative coins are uncorrelated with BTC, so the levered low-beta leg is a bet on idiosyncratic pumpers, and it was lucky in 2021.
  - That long leg is exactly where survivorship bias inflates results.
  - The strategy is net long about 0.2 in dollars and pays 3.2%/yr in funding. Deflated Sharpe probability is 0.21 (n = 86).
  - **Weak.**
- **Flipped sign, B-rev (10 configs + 3 robustness): a "lottery premium".** The best is `BR_SKEWrev_4h30_DN_R1`: long the coins whose 4h returns over 30 days are most positively skewed, short the most negatively skewed.
  - Net Sharpe 0.86 (gross 1.04), 0.67 at 2x cost, 1.00 at lag 1.
  - Plateau median 0.78. The top-30 universe gives 0.62 and q = 0.3 gives 0.89.
  - **2022 Sharpe is -0.41.** Yearly returns: 2020 +40%, 2021 +102%, 2022 -8%, 2023 +8%, 2024 +8%. Most of the PnL comes from the small-universe years 2020-21.
  - Spanning regression on BTC, xsmom14 and 3-day residual momentum: alpha +9%/yr with **t = 0.66**. Loadings are 0.30 on xsmom14 and 0.12 on BTC.
  - The PnL comes from the **short leg**. Coins with negatively skewed returns lag the EW universe by 25%/yr, and that gap is positive in every year: excess Sharpe 3.3, 2.7, 0.4, 1.0, 0.5, shrinking over time. The long "lottery" leg is unstable (2022: -0.95).
  - I tested one post-hoc config: the short leg alone, hedged with long BTC (`famB_shortleg.py`). It fails: Sharpe 0.14, 2021 -52%. The edge exists only *relative to an alt basket*.
  - Deflated Sharpe probability is 0.19.
  - **Weak.**
- **Survivorship.** The 150-coin list contains only coins alive in September 2026.
  - For the textbook B direction, dead lottery coins are missing. That makes the short leg look worse than it really was, so the bias works *against* the anomaly. Even so, the rejection is so broad (28 of 28 configs) that survivorship cannot plausibly rescue it.
  - For BAB-BN's long leg and B-rev's long leg, survivorship flatters the results.
  - B-rev's short leg is conservatively biased. It is the most credible piece found, but it can only be traded against an alt basket, and that basket's returns are themselves inflated by survivorship.

## Implementation notes for the lead
- `lab.quantile_ls_weights` is not symmetric in sign. The percentile cut gives about 8 names long and 7 short when the universe has 40 names. As a result, a flipped score is not the exact negative of the original book.
  - In `chk_flip.py`, flipping skew-30d gives gross Sharpe 0.91, against 1.11 for minus the original book. In 2022 the two disagree in sign (-0.53 vs +0.08).
  - Single-name sensitivity at the quintile edge is large while the universe is small (2020-21 has 16 to 40 names).
- `lab.backtest(rebalance_every=k)` and my own `hold()` keep the *target* weights constant between updates and re-trade drift every day. This is slightly conservative on cost, about 1%/yr.
- Minimum equity at $10 per position: BAB-BN needs about $212 (p90 $285), so it is not feasible at $200. B-rev needs $160 (p90 $200).

## Suggestions (not run, would need fresh pre-registration)
1. The 4h low-volume reversal is real before costs but is a market-making edge. It is only worth revisiting as a *maker-only* execution overlay on another strategy's rebalances, for example skewing limit-order placement toward the reversal side. It is not a standalone taker strategy.
2. "Short crash-skewed coins vs an alt basket" is the most consistent single leg, positive in every year. A cheap test is to use it as a *filter* on existing long books (RMF momentum, the CLR long basket): exclude coins in the most negative skew quintile. That needs no new hedge leg.
3. Treat BAB, IVOL and MAX as regime-dependent, not as alpha. They are 2022 bear-market hedges only, and they lose in every risk-on year.
