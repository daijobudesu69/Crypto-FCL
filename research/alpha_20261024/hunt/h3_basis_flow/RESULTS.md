# h3_basis_flow: results (research data 2020-01 to 2024-12 only, loaded through lab.py)

**Short answer.** One candidate is worth forwarding: **cross-sectional spot-perp BASIS contrarian**
(`A3_basis_lvl_7`). It meets every forwarding criterion in the brief. It is **not**
statistically conclusive: the deflated Sharpe is 0.50 at n=73 trials and the alpha t is about 1.9
after controlling for the funding factor. Treat it as "promising, needs the forward test", not as
a proven edge. The time-series basis/premium configs found nothing. Family B (order flow and
attention) has no edge of its own: its best config is mostly 7d momentum and is flat in 2023-24.

Trials: **73 configs** (A cross-section 19, A time-series 12, A robustness 10, A post-hoc 1,
B 31), every one logged in `registry.csv`. Costs: 0.07%/side plus real Binance funding. "2x"
means 0.14%/side; "lag1" means `extra_lag=1`.

## Family A: basis / premium

**Hypothesis.** A rich perp (perp above spot) means levered longs are paying up. The other side is
retail leverage demand. The trade is short the rich coins and long the coins at a discount
(shorts and hedgers pressing the perp). The premium persists because arbitrage is limited:
cash-and-carry needs capital on both legs, and the reverse carry (short spot to buy a cheap perp)
needs a spot borrow, which is scarce for alts. So a discount reflects real short pressure that
arbs cannot close.

**Grid.** The grid was written in the header of `10_basis_premium.py` before running. It covers
prem_mean level for L=1/3/7/14, prem_mean trailing-90d z for L=1/3/7, basis level for L=1/3/7,
basis z for L=3/7, premium-minus-funding rank for L=3/7, demean weights, rebalance every 3 days,
a funding control, and time series on BTC/ETH/EW basket (2 L values x 2 rules).
Robustness configs are in `30_basis_deepdive.py`.

**Findings**
* **prem_mean contrarian is essentially the trailing-funding factor.** The daily cross-sectional
  Spearman with fund_7 is 0.81 and the return correlation is 0.79. Its Sharpe is about 1.1, but
  all of that comes from 2020-21. 2023-24 is negative (A1_pm_lvl_7: 2.20 / 0.21 / -0.19, 2024
  Sharpe -0.95). The "premium beyond funding" configs (A5) lose -1.2 to -1.3. Inside the funding
  clamp band, prem_mean adds nothing tradable beyond funding. Verdict: weak/none, and the
  funding hunter covers it anyway.
* **Spot-perp basis (perp close / Binance spot close - 1, 7d mean) is different.** Its
  cross-sectional Spearman with fund_7 is only 0.40 and the return correlation with the funding
  control is 0.36. It is positive in every sub-period. Decomposition:
  * The premium-index snapshot component (R1) is 2020-21 heavy, like funding: 2.30 / 0.26 / 0.38.
  * The "Binance spot vs index plus last-trade noise" component (R2) alone has no edge (-0.40).
    The combined basis beats both parts.
  * **Same weights, PnL measured on spot prices:** +26.9%/yr (Sharpe 1.04), versus +34.4% on perp
    prices. With a 1-bar delay the spot PnL is still +26.9% (Sharpe 0.92). So this is not a
    perp close-price bounce artifact. About 7%/yr is mechanical basis convergence.
  * Dropping today's snapshot (R3, basis shifted 1 day) cuts Sharpe to 0.80. The freshest
    reading matters, and the IC decays from 0.021 (day 1) to 0.007 (day 7).
  * PnL source: price +34.6%/yr, funding received +8.2%/yr, cost drag -11.4%/yr
    (turnover about 163x per year).
  * Excess over the EW basket: long leg Sharpe 0.78, short leg 1.25. Most of the edge comes
    from shorting rich-basis coins, which is the side that survivorship bias *understates*.
* **Time series (BTC/ETH/EW basket): none.** The long/short rule ranges from -0.72 to +0.22.
  The long-only "buy the discount" rule is below buy & hold on ETH (0.72 vs 1.00) and on the
  EW basket (0.55 vs 0.96). BTC_L3_b (0.96 vs 0.90 buy & hold, maxDD -18%) is exposure timing
  with BTC correlation 0.65, not alpha.

### Candidate 1, to forward: `A3_basis_lvl_7` (spec frozen)
* **Signal.** `basis = close_perp / spot_close - 1` at the daily 00:00 UTC close (Binance).
  Values with |basis| >= 5% are set to NaN. `s = rolling_mean(basis, 7, min_periods=4)`.
  Score = `-s`.
* **Universe.** `lab.universe('1d', min_hist_days=60, top_k=50)`, restricted to coins with a
  Binance spot close that day (about 88-94% of the universe).
* **Weights.** `lab.quantile_ls_weights(score, U, q=0.2)`: long the bottom-basis quintile, short
  the top quintile, equal weight, dollar neutral, gross 1.0.
* **Timing.** Rebalance daily at the close. No stops; exits happen only through rank changes.
* **IS 2020-03-01 to 2024-12-31:** Sharpe 1.21, CAGR 32.3%, maxDD -29.5%, t_NW 2.70,
  64% positive months.
* **Sub-periods:** 2020-21 1.44, 2022 1.13, 2023-24 1.09. Every calendar year is positive:
  +3%, +123%, +19%, +26%, +13% (yearly Sharpe 0.27 / 2.13 / 1.13 / 1.49 / 0.74).
* **Stress:** 2x cost 0.77, lag1 0.80, 5 best days removed 0.92.
* **Plateau (neighbour median 1.07):** L3 1.11, L14 0.64, z-version 1.35, top-30 1.07,
  top-80 0.99, skip-today 0.80.
* **Correlations:** BTC +0.01, 14d XS-mom +0.06, funding control +0.36. Regression alpha versus
  BTC, mom and fund7: +19.6%/yr, t_NW 1.87.
* **Concentration.** CRV, BNB, ETC, DOGE and ETH make up 83% of gross PnL. Removing the top 3
  coins gives Sharpe 0.84.
* **Size and capital.** About 14 positions on average. Minimum equity is about $200 for every
  position to be at least $10 (95th percentile).
* **$200 path (IS):** $200 grows to $776 by end-2024. The minimum was $184.
* **DSR:** 0.50 at n=73, 0.62 at n=36, 0.72 at n=20 (the trials are heavily correlated, so the
  effective n is lower than 73).

### Alternate: `A4_basis_z_7` (same family; do not forward both)
The score is the trailing-90d z of the 7d basis.
* **Performance:** Sharpe 1.35, CAGR 42%, maxDD -35%.
* **Sub-periods:** 1.24 / 0.79 / 2.24. 2020 is negative (-0.44).
* **Stress:** 2x cost 0.92, lag1 0.70.
* **Plateau:** neighbour median 1.09.
* **Main concerns:** extreme fat tails (skew 4.5, excess kurtosis 89) and a weaker top-30
  result (0.68).
* **Post-hoc combo:** blending the A3 and A4 ranks (C1) *lowered* Sharpe to 0.92. This is a
  fragility warning: the quintile edge depends on the exact definition.

## Family B: order flow and attention

**Hypothesis.** Taker flow is either informed (continuation) or levered noise that liquidity
providers absorb (reversal). Attention shocks lead to overreaction and then reversal. Heavy taker
buying with a flat price means passive sellers are absorbing, so the price should fall. The grid
(31 configs) is in the header of `20_flow_attention.py`: TBR level and z for L=1/3/7 with both
signs, volume and trade-count shock for L=1/3 with both signs, reversal control and
shock-conditioned reversal, flow-price divergence for L=1/3/7, and 4h TBR z.

**Findings.** Nothing here is an edge of its own.
* **TBR continuation.** The best config is `B1_tbr_lvl_7_cont`: Sharpe 1.32, 2x 0.96, lag1 1.09.
  But its correlation with 14d XS momentum is 0.57 (beta 0.47), 2023-24 is only 0.22, and 2023
  is negative. Regression alpha is +14.8%/yr, t 1.38. **Weak: it is mostly a momentum proxy.**
* **Reversal side.** TBR reversal, short-term reversal (1d and 3d) and shock-conditioned
  reversal all lose -1.6 to -2.5. Daily rank IC does say reversal (+0.035) and "high volume
  shock leads to lower returns" (-0.03). But equal-weight dollar quintile returns go the other
  way because of heavy right tails and survivorship, and costs (24-39%/yr) dominate in any case.
* **Volume and trade-count shock.** At most 0.33 (`B3_vshock_3_long`, 2022 negative). None.
* **Flow-price divergence.** -0.2 to -2.2, also negative when reversed. None.
* **4h TBR.** Costs of 86-229%/yr destroy it. None.

## Concerns (be explicit)
1. **Weak significance.** DSR is about 0.5 and alpha t is about 1.9. The IS Sharpe of about 1.2
   is close to the expected maximum of many null trials.
2. **Concentration and size.** The edge relies on a handful of coins (CRV 2022-23, DOGE 2021).
   With q=0.2 and about 45 eligible names there are only 7 coins per side.
3. **Execution timing.** The edge is strongest on the freshest 00:00 UTC basis snapshot, which is
   also the funding timestamp and a thin-liquidity moment. Live execution would happen minutes
   later (the 1-day delay still gives 0.80).
4. **Venue mismatch.** The signal needs Binance spot plus perp data. On Hyperliquid, funding is
   hourly and the coin list differs; the edge after HL fees and funding is untested.
5. **Survivorship.** The 150-coin survivor universe inflates the long leg. For A3 most of the
   excess comes from the short leg, which survivorship understates, so it is less exposed. A4
   leans more on the long leg.

## Suggestions for the lead (not tested; would add trials)
* Rebalancing A3 every 2-3 days could cut the 11%/yr cost drag, since the level IC decays
  slowly. Test it only as a declared extra trial.
* Check whether a basis plus funding blend is merely carry. Its correlation with the funding
  control is only 0.36, so they may diversify, but the C1 result warns against blending on
  hindsight.

Files:
* Scripts: `00_inspect.py`, `h3lib.py`, `10_basis_premium.py`, `20_flow_attention.py`,
  `30_basis_deepdive.py`, `40_candidates.py`
* Logs: `famA.log`, `famB.log`, `deepdive.log`, `candidates.log`
* Data: `registry.csv`, `candidates.csv`

## Appendix: all 73 configs
Lx/Sx = long/short leg Sharpe in excess of the EW top-50 basket. "cost/yr" is the cost drag.
| family | config | Sh | Sh 2x | Sh lag1 | 20-21 | 2022 | 23-24 | Lx | Sx | price/yr | fund/yr | cost/yr | corr MOM | maxDD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A_basis | A0_fund7_ctrl | +1.11 | +0.83 | +0.99 | +1.88 | +0.76 | +0.17 | +1.12 | +0.84 | +25% | +14% | +8% | +0.08 | -45% |
| A_basis | A1_pm_lvl_1 | +0.61 | -0.22 | +0.72 | +1.46 | +0.91 | -1.06 | +1.02 | +0.00 | +26% | +13% | +23% | +0.07 | -47% |
| A_basis | A1_pm_lvl_3 | +1.21 | +0.77 | +0.90 | +2.26 | +0.39 | -0.15 | +1.03 | +1.02 | +33% | +13% | +12% | +0.10 | -34% |
| A_basis | A1_pm_lvl_7 | +1.11 | +0.84 | +1.20 | +2.20 | +0.21 | -0.19 | +1.01 | +0.88 | +25% | +13% | +7% | +0.05 | -41% |
| A_basis | A1_pm_lvl_14 | +1.11 | +0.92 | +1.08 | +1.98 | +1.11 | -0.22 | +1.01 | +0.89 | +23% | +12% | +5% | +0.12 | -46% |
| A_basis | A2_pm_z_1 | +0.86 | -0.04 | +0.56 | +1.77 | +1.39 | -1.06 | +1.21 | +0.22 | +41% | +8% | +25% | +0.10 | -41% |
| A_basis | A2_pm_z_3 | +1.03 | +0.53 | +0.49 | +2.11 | +0.15 | -0.43 | +1.00 | +0.78 | +34% | +7% | +14% | +0.12 | -41% |
| A_basis | A2_pm_z_7 | +0.81 | +0.50 | +0.44 | +1.55 | +0.53 | -0.31 | +0.79 | +0.60 | +25% | +7% | +9% | +0.05 | -32% |
| A_basis | A3_basis_lvl_1 | -0.22 | -1.57 | -0.63 | +0.50 | -0.88 | -1.25 | +0.06 | -0.37 | +22% | +7% | +34% | +0.09 | -75% |
| A_basis | A3_basis_lvl_3 | +1.11 | +0.45 | +0.53 | +1.46 | +0.95 | +0.81 | +1.39 | +0.37 | +42% | +8% | +19% | +0.03 | -37% |
| A_basis | A3_basis_lvl_7 | +1.21 | +0.77 | +0.80 | +1.44 | +1.13 | +1.09 | +0.78 | +1.25 | +35% | +8% | +11% | +0.06 | -30% |
| A_basis | A4_basis_z_3 | +0.87 | +0.16 | +0.94 | +1.14 | -0.30 | +1.15 | +0.94 | +0.50 | +39% | +5% | +20% | +0.09 | -41% |
| A_basis | A4_basis_z_7 | +1.35 | +0.92 | +0.70 | +1.24 | +0.79 | +2.24 | +1.27 | +0.98 | +47% | +4% | +12% | +0.11 | -35% |
| A_basis | A5_pm_xfund_3 | -1.28 | -2.29 | -1.41 | -1.81 | -0.46 | -0.98 | -1.14 | -0.90 | -5% | -2% | +23% | -0.03 | -80% |
| A_basis | A5_pm_xfund_7 | -1.20 | -1.94 | -0.86 | -0.92 | -1.64 | -1.64 | -1.04 | -0.86 | -9% | -2% | +17% | -0.09 | -78% |
| A_basis | A6_pm_lvl_7_demean | +1.26 | +1.07 | +1.13 | +2.56 | +0.49 | -0.22 | +1.06 | +1.11 | +25% | +19% | +6% | +0.06 | -50% |
| A_basis | A6_pm_z_7_demean | +1.04 | +0.78 | +0.81 | +2.10 | -0.03 | -0.18 | +1.03 | +0.70 | +25% | +9% | +7% | +0.10 | -36% |
| A_basis | A7_pm_lvl_7_rebal3 | +1.17 | +0.97 | +1.39 | +2.24 | +0.55 | -0.24 | +1.17 | +0.82 | +26% | +11% | +5% | +0.03 | -43% |
| A_basis | A7_pm_z_7_rebal3 | +0.53 | +0.28 | +0.82 | +1.33 | +0.24 | -0.67 | +0.40 | +0.51 | +14% | +6% | +6% | +0.07 | -40% |
| A_basis_TS | ATS_BTC_L3_a | +0.22 | +0.14 | +0.09 | +0.24 | +1.77 | -0.65 | +0.05 | +1.23 | +5% | +6% | +3% | -0.13 | -46% |
| A_basis_TS | ATS_BTC_L3_b | +0.96 | +0.91 | +0.82 | +1.15 | +0.28 | +1.30 | +0.05 | +nan | +26% | -1% | +1% | -0.13 | -18% |
| A_basis_TS | ATS_BTC_L7_a | +0.12 | +0.07 | +0.05 | +0.39 | +0.79 | -0.62 | +0.20 | +1.18 | +1% | +5% | +2% | -0.15 | -50% |
| A_basis_TS | ATS_BTC_L7_b | +0.85 | +0.82 | +0.87 | +1.39 | -0.67 | +1.14 | +0.20 | +nan | +24% | -1% | +1% | -0.13 | -26% |
| A_basis_TS | ATS_ETH_L3_a | -0.05 | -0.11 | -0.23 | +0.12 | -0.03 | -0.37 | +0.38 | +1.20 | -6% | +6% | +3% | -0.14 | -55% |
| A_basis_TS | ATS_ETH_L3_b | +0.72 | +0.69 | +0.63 | +1.22 | -0.32 | +0.84 | +0.38 | +nan | +27% | -1% | +1% | -0.15 | -37% |
| A_basis_TS | ATS_ETH_L7_a | -0.38 | -0.41 | -0.52 | +0.00 | -1.04 | -0.63 | +0.33 | +0.97 | -22% | +6% | +2% | -0.15 | -82% |
| A_basis_TS | ATS_ETH_L7_b | +0.64 | +0.62 | +0.61 | +1.35 | -0.94 | +0.82 | +0.33 | +nan | +24% | -2% | +1% | -0.15 | -47% |
| A_basis_TS | ATS_EW_L3_a | -0.48 | -0.54 | -0.71 | -0.78 | +0.45 | -0.66 | -9.1 | -5.5 | -29% | +8% | +3% | -0.23 | -86% |
| A_basis_TS | ATS_EW_L3_b | +0.55 | +0.52 | +0.33 | +0.78 | +0.05 | +0.77 | -9.1 | +nan | +21% | -0% | +1% | -0.20 | -41% |
| A_basis_TS | ATS_EW_L7_a | -0.72 | -0.76 | -0.89 | -1.04 | +0.19 | -0.88 | -7.4 | -4.39 | -43% | +8% | +2% | -0.23 | -93% |
| A_basis_TS | ATS_EW_L7_b | +0.38 | +0.36 | +0.20 | +0.60 | -0.33 | +0.76 | -7.4 | +nan | +15% | -0% | +1% | -0.20 | -49% |
| B_flow | B1_tbr_lvl_1_cont | -0.44 | -1.79 | -0.19 | +0.71 | -1.28 | -2.03 | -0.37 | -0.30 | +19% | +3% | +32% | +0.45 | -66% |
| B_flow | B1_tbr_lvl_1_rev | -2.28 | -3.57 | -2.80 | -2.85 | -2.22 | -1.71 | -2.39 | -1.52 | -21% | -4% | +32% | -0.44 | -95% |
| B_flow | B1_tbr_lvl_3_cont | +0.81 | +0.17 | +0.78 | +1.75 | +0.10 | -0.27 | +0.91 | +0.43 | +34% | +4% | +17% | +0.48 | -36% |
| B_flow | B1_tbr_lvl_3_rev | -2.25 | -2.85 | -2.36 | -2.75 | -1.72 | -2.00 | -1.96 | -1.81 | -41% | -4% | +17% | -0.45 | -96% |
| B_flow | B1_tbr_lvl_7_cont | +1.32 | +0.96 | +1.09 | +2.16 | +1.18 | +0.22 | +1.26 | +0.97 | +41% | +4% | +10% | +0.57 | -28% |
| B_flow | B1_tbr_lvl_7_rev | -2.15 | -2.50 | -1.83 | -2.88 | -1.92 | -1.37 | -1.75 | -1.84 | -46% | -4% | +10% | -0.56 | -96% |
| B_flow | B2_tbr_z_1_cont | -0.13 | -1.56 | -0.29 | +1.76 | -1.85 | -2.50 | -0.08 | -0.09 | +30% | +2% | +35% | +0.33 | -74% |
| B_flow | B2_tbr_z_1_rev | -2.53 | -3.89 | -2.65 | -3.62 | -1.57 | -1.58 | -2.61 | -1.74 | -28% | -3% | +35% | -0.33 | -97% |
| B_flow | B2_tbr_z_3_cont | +0.79 | +0.05 | +0.75 | +1.92 | -0.95 | -0.16 | +0.91 | +0.39 | +38% | +3% | +19% | +0.45 | -44% |
| B_flow | B2_tbr_z_3_rev | -2.48 | -3.18 | -2.48 | -3.34 | -0.99 | -2.13 | -2.29 | -1.96 | -46% | -3% | +19% | -0.40 | -97% |
| B_flow | B2_tbr_z_7_cont | +0.99 | +0.60 | +0.66 | +1.74 | +0.49 | +0.04 | +1.25 | +0.47 | +39% | +3% | +12% | +0.46 | -35% |
| B_flow | B2_tbr_z_7_rev | -1.68 | -2.08 | -1.55 | -2.36 | -1.10 | -1.08 | -1.12 | -1.63 | -35% | -3% | +12% | -0.45 | -94% |
| B_flow | B3_vshock_1_long | +0.30 | -0.32 | +0.02 | +0.74 | -0.98 | +0.13 | +0.41 | +0.15 | +27% | +2% | +20% | +0.54 | -50% |
| B_flow | B3_vshock_1_short | -1.72 | -2.35 | -1.31 | -2.04 | -0.91 | -1.90 | -1.34 | -1.58 | -33% | -3% | +20% | -0.55 | -95% |
| B_flow | B3_vshock_3_long | +0.33 | -0.03 | +0.42 | +0.48 | -0.36 | +0.43 | +0.57 | +0.03 | +20% | +2% | +12% | +0.61 | -44% |
| B_flow | B3_vshock_3_short | -1.38 | -1.74 | -1.29 | -1.45 | -0.72 | -1.88 | -0.99 | -1.33 | -31% | -3% | +12% | -0.62 | -93% |
| B_flow | B4_tshock_1_long | +0.06 | -0.57 | -0.39 | +0.40 | -1.17 | -0.05 | +0.35 | -0.22 | +20% | +2% | +20% | +0.51 | -54% |
| B_flow | B4_tshock_1_short | -1.33 | -1.95 | -1.16 | -1.41 | -0.58 | -1.84 | -0.81 | -1.39 | -20% | -3% | +20% | -0.51 | -91% |
| B_flow | B4_tshock_3_long | +0.11 | -0.25 | +0.38 | +0.21 | -0.99 | +0.44 | +0.44 | -0.22 | +13% | +2% | +12% | +0.58 | -47% |
| B_flow | B4_tshock_3_short | -1.13 | -1.49 | -1.10 | -1.16 | -0.46 | -1.62 | -0.83 | -1.04 | -22% | -2% | +12% | -0.58 | -88% |
| B_flow | B5_rev_1_ctrl | -1.77 | -2.97 | -1.41 | -2.42 | -0.88 | -1.39 | -1.63 | -1.45 | -18% | -1% | +39% | -0.31 | -97% |
| B_flow | B5_shockrev_1 | -1.69 | -2.88 | -1.73 | -2.04 | -2.10 | -1.16 | -1.48 | -1.40 | -15% | -1% | +37% | -0.30 | -96% |
| B_flow | B5_rev_3_ctrl | -1.57 | -2.23 | -1.32 | -2.03 | -0.17 | -1.77 | -0.95 | -1.66 | -32% | -1% | +24% | -0.46 | -96% |
| B_flow | B5_shockrev_3 | -1.88 | -2.58 | -1.74 | -2.82 | +0.36 | -1.71 | -1.46 | -1.68 | -37% | -1% | +22% | -0.51 | -96% |
| B_flow | B6_div_1 | -2.21 | -3.92 | -1.86 | -2.21 | -1.89 | -2.49 | -2.33 | -1.19 | -10% | -2% | +39% | +0.04 | -93% |
| B_flow | B6_div_3 | -0.91 | -1.93 | -1.66 | -0.41 | -2.28 | -1.04 | -0.97 | -0.52 | +5% | -2% | +25% | +0.12 | -75% |
| B_flow | B6_div_7 | -0.20 | -0.94 | -0.85 | -0.10 | -0.81 | +0.00 | -1.00 | +0.56 | +15% | -2% | +18% | +0.33 | -58% |
| B_flow | B7_4h_tbr_z_1_cont | -8.83 | -18.34 | -9.02 | -4.89 | -13.8 | -14.6 | -6.2 | -8.6 | +18% | +1% | +229% | +0.19 | -100% |
| B_flow | B7_4h_tbr_z_1_rev | -10.50 | -20.13 | -10.76 | -9.4 | -10.0 | -14.8 | -9.6 | -8.2 | -23% | -2% | +229% | -0.17 | -100% |
| B_flow | B7_4h_tbr_z_6_cont | -1.55 | -4.84 | -1.17 | +1.10 | -5.9 | -4.42 | -0.38 | -2.56 | +43% | +2% | +86% | +0.31 | -94% |
| B_flow | B7_4h_tbr_z_6_rev | -5.12 | -8.42 | -5.34 | -5.9 | -3.32 | -5.7 | -4.48 | -4.28 | -47% | -3% | +87% | -0.29 | -100% |
| A_basis_robust | R1_prem_snap_7 | +1.36 | +0.93 | +0.61 | +2.30 | +0.26 | +0.38 | +1.22 | +1.08 | +39% | +10% | +12% | +0.07 | -29% |
| A_basis_robust | R2_basis_x_prem_7 | -0.40 | -0.91 | -0.97 | -0.78 | +0.82 | -0.39 | -0.24 | -0.33 | -0% | +3% | +13% | -0.05 | -62% |
| A_basis_robust | R3_basis_lvl_7_skip | +0.80 | +0.41 | +0.61 | +0.77 | +1.11 | +0.96 | +0.75 | +0.63 | +27% | +7% | +11% | -0.01 | -35% |
| A_basis_robust | R4_basis_z_7_skip | +0.76 | +0.32 | +0.34 | +0.68 | +0.79 | +1.14 | +0.78 | +0.49 | +30% | +4% | +12% | +0.09 | -40% |
| A_basis_robust | R5_basis_lvl_7_top30 | +1.07 | +0.66 | +0.56 | +1.29 | +1.53 | +0.62 | +0.85 | +0.89 | +33% | +9% | +12% | +0.06 | -38% |
| A_basis_robust | R6_basis_z_7_top30 | +0.68 | +0.26 | +0.13 | +0.97 | +0.78 | +0.25 | +0.48 | +0.67 | +29% | +4% | +13% | +0.09 | -35% |
| A_basis_robust | R7_basis_lvl_14 | +0.64 | +0.36 | +0.61 | +0.73 | +0.57 | +0.67 | +0.77 | +0.35 | +17% | +8% | +8% | +0.07 | -40% |
| A_basis_robust | R8_basis_z_14 | +1.11 | +0.77 | +0.85 | +1.30 | +0.65 | +1.21 | +0.77 | +1.03 | +33% | +4% | +9% | +0.11 | -37% |
| A_basis_robust | R9_basis_lvl_7_top80 | +0.99 | +0.55 | +0.74 | +1.44 | +1.13 | +0.29 | +0.60 | +1.04 | +29% | +8% | +11% | +0.08 | -30% |
| A_basis_robust | R10_basis_z_7_top80 | +1.09 | +0.66 | +0.45 | +1.24 | +0.79 | +1.33 | +1.12 | +0.67 | +39% | +4% | +12% | +0.11 | -35% |
| A_basis_posthoc | C1_basis_combo_7 | +0.92 | +0.44 | +0.38 | +0.88 | +1.39 | +1.12 | +0.95 | +0.59 | +34% | +7% | +14% | +0.07 | -38% |
