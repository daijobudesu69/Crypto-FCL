# r2_vol_structure: round 2, HYPE-candle-only (IS 2020-01-01 to 2024-12-31)

**Bottom line**
- **Family A (asymmetric betas and coskewness): none.** The textbook directions are rejected. The one cell that clears the bar is BAB (betting against beta) in disguise, which round 1 already covered.
- **Family B (volatility structure): one candidate.** It is **B_pkcc30_FLIP_R1**: long coins whose 30-day Parkinson vol is LOW relative to their close-to-close vol, short coins where it is HIGH. Net Sharpe is 1.34, every sub-period is positive, 2x cost gives 1.09, lag 1 gives 0.92, and alpha against xsmom14 plus BTC has t = 2.24.
- I rate it **promising, but marginal**. It is a fast signal: holding it 5 to 7 days kills it. The pre-registered neighbours are weak. The 16-position version has Sharpe 1.0, and its lag-1 Sharpe is 0.55.

**Protocol**
- All data came through `lab.py`. `ALPHA_FORWARD` was never set, and no 2025+ data or results were read.
- Signal inputs are `open`, `high`, `low` and `close` (1d), plus 4h `close` for the term-structure realized variance (RV). The universe ranking uses `qv` through `lab`. No funding, OI, long/short, premium, spot or taker data is used as an input.
- Funding is charged as a cost. Cost is `lab.HYPE_COST` (0.065% per side).
- Universe is `lab.hype_universe('1d', top_k=50)`. Family A excludes BTC, because its beta is 1 by construction.
- Trials:
  - Family A: 32 grid configs + 4 diagnostics = 36.
  - Family B: 32 grid configs + 17 robustness configs = 49.
  - Hunter total: 85.
  - Every grid is pre-registered in its script header.

## Family A: asymmetric risk premia (`famA_grid.py`, `famA_diag.py`, `famA_grid.csv`)

**Hypothesis.** The Ang-Chen-Xing (ACX) downside-beta premium: long high beta-on-BTC-down-days. For upside beta, the textbook direction is long low β+. For coskewness (Harvey-Siddique), it is long negative coskew. All are measured against BTC over 90d or 180d.

**Grid.** 4 signals (β−, β+, β−−β+, coskew) × 2 windows × 2 directions × 2 constructions = 32. The constructions are $-neutral quintiles and beta-neutral legs. All configs rebalance weekly.

**Results.**
- **The textbook directions fail.** The best textbook config is β− 180 DN, with Sharpe 0.52. Its sub-periods are 2022 −0.14 and 2023-24 −0.17, and all of its PnL comes in 2021.
- **Coskewness has no edge.** The best is 0.40, with 2023-24 at −0.29.
- **The asymmetry β−−β+ has no edge** in either direction (|SR| ≤ 0.52, with sub-periods of mixed sign).
- **The best cell is `A_bminus_90_FLIP_BN`**: long LOW downside beta, beta-neutral. Sharpe 1.17, 2x cost 1.09, lag 1 1.17. Sub-periods are 1.77, 0.36 and 0.75. Alpha against xsmom14 plus BTC has t = 1.97. It is rated **weak**, for these reasons:
  - It is BAB. The cross-sectional Spearman correlation of β− with β is 0.59. Its returns load 0.35 on BAB-BN(90) with t = 7.3. Residual alpha has t = 1.09.
  - A pure-downside component, β− residualized on β, gives Sharpe 0.30 at 90d and −0.52 at 180d.
  - The 180d twin has Sharpe −0.29, so the plateau median is about 0.12.
  - The top 5 coins are XRP, BNB, FIL, XMR and TRX. They make up 97% of price PnL.
  - The long leg does not beat EW (excess Sharpe 0.08).
  - It is net long about +0.19 in dollars, and its correlation with BTC is 0.22.
- **Verdict for the family: none.**

## Family B: cross-sectional volatility structure (`famB_grid.py`, `famB_robust.py`, `famB_diag2.py`)

**Grid.** 8 signals × 2 directions × 2 rebalance frequencies (1d or 7d) = 32. All are $-neutral quintiles.
- vts5 and vts10: realized vol over the short window / realized vol over 60d, using intraday RV built from 4h bars.
- ATR5/50 and ATR20/100.
- Vol-of-vol 30 and 90: std of log(log H/L).
- PK/CC 30 and 90: Parkinson vol / close-to-close vol.

**Results.**
- **Compression vs expansion (vts, ATR).** Long compressed loses every time (as low as −1.27 at R1).
  - The flip (long expanded) gives 0.2 to 0.57, but its correlation with xsmom14 is 0.35 to 0.52 and its alpha t is ≤ 0.07. That is momentum: in alts, vol expands on up-moves. **None.**
- **Vol-of-vol.** The textbook direction (long low VoV) loses.
  - The flip (long high VoV) gives 0.60 to 0.69, but 2022 is negative (−0.28 to −0.65) and kurtosis is 97. This is the round-1 "lottery wins in bull years" pattern. **None to weak.**
- **PK/CC.** The Corwin-Schultz illiquidity direction (long wicky/noisy coins) loses at −1.70 (R1).
  - The **flip is the candidate.**

### Candidate: B_pkcc30_FLIP_R1

**Rules.**
- Signal: `ratio = sqrt(mean30[(ln H/L)^2/(4 ln2)] / mean30[(ln C_t/C_t-1)^2])`, with at least 24 observations.
- Long the bottom 20% of the HYPE top-50 universe by `ratio`, short the top 20%.
- Legs are equal-weight, 0.5 gross each. The book is $-neutral with gross 1.0.
- The book is rebuilt at every 00:00 UTC daily close. There are no stops.

**Headline metrics.**

| Metric | Value |
|---|---|
| Sharpe (net) | 1.34 (gross 1.53) |
| CAGR | +39% |
| Vol | 27% |
| MaxDD | −28% |
| Newey-West t | 3.13 |
| Turnover | 106x per year |
| Cost drag | 6.9% per year |
| Funding | +1.7% per year (received) |

**Yearly returns.** 2020 +65%, 2021 +31%, 2022 +15%, 2023 +5%, 2024 +86%.

**Sub-period Sharpe.** 2020-21 1.36, 2022 0.75, 2023-24 1.82.

**Stress.**
- 2x cost: 1.09. 3x cost: 0.83.
- Extra lag 1 day: 0.92. Lag 2: 0.17. Lag 3: −0.09.
- Execution delay measured on 4h bars: 4h 1.18, 8h 1.12, 12h 1.07. A GitHub Actions cron that runs 5 to 30 minutes late is harmless.

**Plateau (Sharpe).**

| Variation | Values tested | Sharpe |
|---|---|---|
| Window | 10 / 20 / **30** / 45 / 60 / 90 | 0.66 / 1.22 / **1.34** / 1.52 / 0.64 / 0.23 |
| Rebalance (days) | **1** / 2 / 3 / 5 / 7 | **1.34** / 1.21 / 0.95 / 0.40 / 0.33 |

- The median over all 9 variations is 0.66.
- The immediate neighbours (20d, 45d, R2) are 1.21 to 1.52.
- The two pre-registered neighbours (R7 and 90d) are only 0.33 and 0.23.

**Recency.**
- Using only the last 5 days of the ratio gives Sharpe 1.71 (gross 2.21), but lag 1 drops to 0.50.
- Using the 30d ratio lagged 5 days gives 0.05.
- So the information sits in the last few days and decays within 2 to 3 days. Signal rank autocorrelation is 0.95 at 1d and 0.79 at 5d.

**Universe robustness.**

| Variant | Sharpe | Lag 1 | Note |
|---|---|---|---|
| HYPE top 30 | 1.13 | 0.75 | |
| HYPE top 75 | 1.33 | | |
| Binance top 50 | 1.11 | | |
| Drop DOGE, BCH, CRV (top PnL coins) | 0.97 | | 2021 −18% |
| Excluding degenerate-range days | 1.33 | | |

**Neutralizations.**
- Beta-neutral: 0.65. The long leg tilts toward higher beta (XS ρ −0.22), and the beta-neutral book ends up net short in dollars.
- Vol-neutral residual: 0.78.

**Book size.**

| Book | Sharpe | 2x cost | Lag 1 | Alpha t | Sub-periods | Min equity |
|---|---|---|---|---|---|---|
| 16-pos (8 long / 8 short) | 1.01 | 0.74 | 0.55 | 1.39 | 0.61 / 1.11 / 1.52 | $160 |
| 12-pos (6/6) | 1.06 | | | | | |

The 2020-21 gap between the 16-pos book and the quintile book exists because 2020 had only about 13 names.

**$200 account simulation (16-pos).** Units are held between changes, there is no drift trading, every order is at least $10, and cost and funding are charged. Result: $200 grows to $637, Sharpe 0.92, MaxDD −44%. By year: 2021 −4%, 2023 0%.

**Legs.**
- Long leg minus EW is positive in every year (+0.18, +0.17, +0.11, +0.13, +0.36 per year in price PnL).
- Short names underperform EW in 4 of 5 years (2023 is about 0).
- Excess Sharpe vs EW: long 0.99, short 1.70.
- Raw leg Sharpe: long +1.26, short −0.52.

**Signed check.**
- Among 5d winners, the low-minus-high PK/CC spread is +23.6 bp/day (t 2.0).
- Among 5d losers it is +13.3 bp/day (t 1.6).
- So the effect is not just signed continuation.

**Momentum overlap.**
- Correlation with xsmom14 is 0.29 and with BTC 0.09.
- Spanning regression on xsmom14 + BTC + 1d/3d/5d continuation + CLV5 + low-vol + 4h-skew30 + BAB-BN: alpha is about +23% per year with t of about 2.2.
- Loadings are xsmom14 0.19 to 0.25 and low-vol −0.12. It is **not mostly momentum**.

**Concentration and sizing.**
- The top 5 coins (DOGE, BCH, CRV, XLM, BNB) give 56% of price PnL, and the top 10 give 88%.
- Rolling 365d Sharpe: minimum −0.47, positive 89% of the time.
- Minimum equity: median $140, p90 $220 for the quintile book; $160 for the 16-pos book.

**Deflated Sharpe probability.** 0.67 with n = 49 (family) and 0.59 with n = 85 (hunter). For the 16-pos book: 0.39 and 0.31.

**Rationale (uncertain, partly post hoc).**
- A low PK/CC ratio means the coin's recent days closed near their extremes with small wicks. Those are "efficient", directional moves, consistent with informed or persistent order flow that liquidity providers did not absorb.
- A high ratio means wicky, choppy days: stop hunts, liquidation spikes, noise flow that got absorbed.
- The wicky coins then lag over the next 1 to 2 days, and the clean movers lead.
- This fits round 1's finding that high-volume moves continue and low-volume moves revert.
- Who is on the other side: retail fading clean moves, and liquidity-takers chasing wicky coins.
- Persistence is questionable. The edge decays in about 2 days, so it is a fast microstructure-flavoured effect, not a risk premium.

**Survivorship.**
- The long leg (clean movers) is inflated by survivor coins.
- The short leg (wicky coins) is understated, because dead wicky coins are missing. That is conservative.
- The larger excess comes from the short leg, which slightly reassures.

**Deployment.**
- The signal needs only the HYPE daily candles: H, L, C and the previous close, 31 days of history.
- It runs as one GitHub Actions job daily after 00:00 UTC. No VPS is needed and no Binance data is needed.
- The live universe ranking would use HYPE volume, not Binance `qv`. This is a small spec difference that the lead should freeze.
- Roughly 4 of the 16 names change per day, with orders of about $12.50.

**Concerns.**
- The holding-period sensitivity (R5 0.40, R7 0.33) and the weak pre-registered neighbours.
- The 2023 near-zero year. A large share of 2023-24 comes from 2024.
- The 16-pos lag-1 Sharpe is 0.55, below 0.6.
- Dependence on DOGE, BCH and CRV in 2021.
- Deflated Sharpe probability is only about 0.6 to 0.67.

## Files
- `common.py`: helpers (metrics, leg decomposition, Newey-West spanning regression, min equity).
- `signals.py`: signal definitions.
- `famA_grid.py`, `famA_diag.py`, `famA_grid.csv`: family A.
- `famB_grid.py`, `famB_robust.py`, `famB_diag2.py`, `famB_grid.csv`, `famB_robust.csv`: family B.
- `final_numbers.py`, `final_numbers.json`: frozen candidates.
- `candidates_daily_IS.parquet`: IS daily returns.
