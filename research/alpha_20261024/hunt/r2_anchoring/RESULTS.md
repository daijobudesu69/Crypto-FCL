# r2_anchoring - round 2 (OHLCV-only signals), IS 2020-01-01 .. 2024-12-31

Rules kept: data via lab.py only (no ALPHA_FORWARD, no raw files, no 2025+ numbers), signal inputs =
high/low/close only (universe ranking uses qv inside lab), universe `lab.hype_universe("1d", top_k=50)`
(15-21 names in 2020, ~50 in 2024), cost `lab.HYPE_COST` 0.065%/side + real funding as cost, signal at
close t traded at close t. Grids are written in the script headers before running.
Trials: family A = 45 (24 grid + 21 follow-up/stress variants), family B = 46 (42 grid + 4).

## Family A - anchoring / 52-week-high (famA_grid.py, famA_followup.py, followup2.py, diag_rp.py, span_short.py, final_numbers.py)

Grid (24): score {PH = close/maxHigh_N, LOW = log close/minLow_N, RP = position in N-day range,
PHR = PH residualised cross-sectionally on the N-day return} x N {90,180,365} x rebal {1,7}; quintile L/S.

| score | N=90 rb1 | rb7 | N=180 rb1 | rb7 | N=365 rb1 | rb7 |
|---|---|---|---|---|---|---|
| PH (George-Hwang) | 0.55 | 0.01 | 0.39 | 0.07 | 0.60 | 0.12 |
| LOW | 0.64 | 0.28 | 0.48 | 0.30 | -0.19 | -0.02 |
| RP | **0.96** | 0.30 | 0.73 | 0.27 | **0.84** | 0.25 |
| PHR (anchoring net of momentum) | 0.03 | -0.11 | 0.35 | 0.04 | 0.48 | 0.49 |

Findings
- The canonical 52-week-high (PH365): Sharpe 0.60, 2022 -0.36. Alpha t vs xsmom14+BTC = 0.95. **It is momentum.**
- The pure anchoring test (PHR = nearness to high net of the same-window return) is weak: best 0.49,
  DSR 0.10, alpha t vs xsmom14 0.90. Anchoring adds nothing measurable once you control for past return.
- RP (range position) is the only score that works, and only with **daily** rebalancing: weekly drops
  it from 0.84 to 0.25, and lag 1 halves it. So the edge is short-horizon (7-14d) continuation, not
  anchoring. Spanning confirms it: alpha t vs 14d momentum is 1.1-1.3 (quintile book) and 1.5-1.9
  (8L/8S book), vs 7d mom 1.4-1.9; vs 90d mom 2.3-3.0. R2 vs xsmom14+BTC is only 10-12%, and the
  appraisal ratio is 0.57-0.72, so a modest independent part exists. But you cannot separate it from
  momentum at t>2 with 5 years of data.
- Tail risk: the quintile RP365 book lost 40% in one day (DOGE +390% short squeeze, 2021-01-28). Skew is -6.
  A resting short stop at +30% from the prior close (1% slippage, re-entry next close) fixes it. The stop
  is a post-hoc choice made after seeing that event. Stop levels of 20/30/40% give 1.30/1.16/1.15.

Best candidate (A2): RP365, 8 long / 8 short, daily rebalance, +30% resting stop on shorts.
Sharpe 1.16 (t_NW 2.63), CAGR 30.5%, MaxDD -21%. Sub-periods 1.69 / **0.11** / 1.01. 2x cost 0.97,
lag-1 0.90. Neighbour median 0.99 (RP90/RP180 same book: 1.02/0.97). Long-leg vs EW 0.94,
short-leg vs EW 1.01. corr BTC -0.06, corr xsmom14 0.38. Alpha t vs xsmom14+BTC **1.93**, vs HYPE-14d mom
1.77. DSR (45 trials) 0.53. Turnover 71x/yr, cost 4.8%/yr, funding +2.0%/yr (received).
Min equity $160 (16 x $10). top-30 universe 1.06, top-75 1.30. No stop: 0.98 (lag1 0.75).
Yearly: 2020 +16%, 2021 +102%, 2022 +0%, 2023 +26%, 2024 +23%.
**Verdict: weak.** Robust and deployable, but it is mostly short-horizon momentum, which the user already
runs (RMF/RNT). 2022 is flat. 2021 alt-season dominates. The stop was tuned after the fact. Forward-test it
only as a momentum-adjacent diversifier, not as a new alpha.

## Family B - residual momentum and frog-in-the-pan (famB_grid.py, followup2.py)

Grid B(i) (24): residual vs BTC (beta = trailing 90d), L {30,60,90} x skip {3,7} x {raw, vol-scaled}
x rebal {1,7}, beta-neutral quintile. Grid B(ii) (18): {SGN = %pos-%neg, FIP double sort, IDS = momentum-neutral
ID spread} x L {30,60,90} x rebal {1,7}, skip 3.
- Residual momentum: best RESID L30 skip7 raw rb7 has Sharpe 0.71, but all of it comes from 2020-21 (1.94).
  2022 is -1.40 and 2023-24 is -0.12. 2022 is negative in all 24 configs. It does beat matched plain momentum
  (alpha t 2.12 vs 30d/skip-7 mom, 2.21 vs the beta-neutral version), but vs xsmom14 it is 0.87.
  The 8L/8S version gives 0.26. **none.**
- Frog-in-the-pan: FIP double sort best 0.39 (2022 -0.83), alpha t vs xsmom14 0.06. SGN is around 0.
  The momentum-neutral ID spread is negative in all 6 configs (-0.14 to -0.71). Continuous moves do NOT
  continue more than jumps in crypto. **none.**

## Bottom line
No new alpha distinct from momentum in either family. The only robust thing (RP365 daily, 16 names +
short stop, Sharpe 1.16) is 7-14d continuation in another form (alpha t vs xsmom14 1.9).
Survivorship: the near-high long leg is inflated and the far-below-high short leg is understated
(dead coins are missing).
