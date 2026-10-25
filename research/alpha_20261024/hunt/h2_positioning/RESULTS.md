# h2_positioning: results (research window 2020-01 to 2024-12 only, via lab.py)

Hunter: h2_positioning. Families: (A) smart-vs-retail positioning, (B) open-interest dynamics and taker flow.
All runs are logged in `runlog.csv`: 47 unique family-A configs (one duplicate name) and 35 family-B
configs, 82 in total, plus 4 controls/references. The runs used 1d panels only. I skipped 4h because
both surviving signals are slow (14-day smoothing, 30-45x turnover a year) and every fast OI or flow
signal already lost money at 1d.

## Coverage. The real sample is 2022-2024, and only 2023-2024 for top-trader fields

Average number of names per day inside the PIT top-50 universe that have the field:

| field | 2020 | 2021 | 2022 | 2023 | 2024 |
|---|---|---|---|---|---|
| global_ls | 0.3 | 4.1 | 37.8 (from Feb) | 48.1 | 50 |
| top_pos_ls / top_acct_ls | 0.3 | 4.1 (Dec only) | 5.3 (only Jun and Dec 2022) | 48.1 | 50 |
| oi_usd | 0.3 | 4.3 (Dec) | 39.8 | 48.1 | 50 |
| taker_ls_vol (metrics) | 0.3 | 4.1 | 26.3 (from May) | 48.1 | 50 |
| tbqv/qv (kline taker share) | 14.3 | 35.1 | 40.5 | 48.6 | 50 |

Days with fewer than 20 valid names are flat. The "2021" and "2022" numbers for top-trader signals
therefore come from 25 and 46 active days.

## Key structural finding (applies to all of family B)
Short-horizon contrarian signals have a positive cross-sectional rank IC but lose money as an
equal-weight quintile L/S. Examples: -ret7 has IC1 +0.037 (t 6.7) and -dOI7 has IC1 +0.037 (t 6.7).
The recent winners, OI builders and taker-buy-heavy coins have the lowest median next-day return
(-22 to -28 bp) but the highest mean (+10 to +33 bp), because their right tail is very fat (p99 about
+25%/day). Shorting them pays for the median and loses on the tail (`diag_quintiles.py`). In this
universe a rank IC does not mean a tradable spread.

## Family A: smart-vs-retail positioning
Hypothesis (pre-registered): retail account crowding (global_ls) is contrarian, and the top-trader
position tilt (top_pos_ls) is informed.
- Time-series z-scores (30/90d) and 1/3/7d changes failed: 10 configs, Sharpe -1.95 to +0.51, and
  every one is negative at 2x cost. 1d changes are mostly short-term reversal and turnover destroys them.
- The top-trader "smart money" hypothesis is rejected and in fact reversed. The z-score version
  (zT90) has a Sharpe of -1.26. The level version (A9, mean14 log top_pos_ls) has -1.54 (2023 -2.37,
  2024 -1.76). Top-minus-retail spreads (A3, A5, A8) carry no information.
- The level of retail account crowding works in the pre-registered direction. A7 is -log(global_ls)
  as a cross-sectional level. Raw it scores 0.88. Smoothed over 14 days it scores 1.06.

### Candidate A9f: top-trader crowding contrarian (post-hoc sign flip, so treat with caution)
- **Rules:** score_t = -mean_14d(log top_pos_ls). Universe = lab.universe('1d', top_k=50). Long the
  top quintile and short the bottom quintile, equal weight, dollar-neutral, gross 1x. Rebalance daily
  at 00:00 UTC. Flat if fewer than 20 names have a score.
- **IS (2021-12 to 2024-12):**
  - Headline: Sharpe 1.33, CAGR 18.7%, MaxDD -13.8%, t_NW 2.13.
  - **2023-24 only:** Sharpe 1.74, CAGR 30.5%, t_NW 2.34.
  - By year: Dec-2021 fragment -10.1% (25 days), 2022 fragments +10.4%, 2023 +30.1% (Sharpe 2.11), 2024 +31.0% (Sharpe 1.54).
  - Half-years 2023H1 3.07, 2023H2 1.12, **2024H1 -0.17**, 2024H2 2.93. 2024Q4 alone made +31%; excluding it, 2023-01 to 2024-09 has a Sharpe of 1.08.
- **Stress tests:**
  - Cost and lag: 2x cost 1.17, 3x cost 1.01. Lag 1, 2 and 3 give 1.40, 1.26 and 0.94.
  - Plateau: S7 0.99, S30 1.19, S14 with weekly rebalance 1.22, top-100 1.37, raw S1 0.24. The neighbour median is about 1.1.
  - Concentration (2023-24): leave-one-coin-out minimum 1.15. Dropping the top-10 contributors leaves 0.83.
- **PnL source:**
  - Price vs funding: per year about +24-27% gross price, +5% funding received (it shorts high-funding coins), -2.5% cost.
  - Long vs short leg: the short leg drives it. Short-leg excess vs the EW universe is +41.9%/yr (Sharpe 2.15); long-leg excess is +5.0%/yr (0.25). Survivorship bias understates short-side results, so this is conservative on that side.
- **Is it just funding?** Its cross-sectional correlation with the trailing 30d funding level is 0.53. Fund7-neutralised, it still scores 0.81. In a spanning regression on the fund7 and fund30 quintile factors (2023-24), alpha is +26.6%/yr with NW-t 2.54. Beta to fund30 is 0.32, and the funding factors themselves scored only about 0.2 in 2023-24. So funding explains part of it but not all.
- **Diversification and size:**
  - Correlations: BTC -0.14, 14d XS momentum 0.14, A7 0.26.
  - Positions and capital: about 20 positions in 2023-24 (10 a side). Minimum equity at $10 per position is about $220. $200 grew to $339 over the whole IS period.
- **Deflated Sharpe:** 0.64 at n=47 (family A), 0.57 at n=82 (all), 0.81 / 0.77 for the 2023-24 window.
- **Verdict:** promising, but post-hoc. The pre-registered direction ("top traders are informed")
  failed, and I flipped it after seeing the result.
- **Ex-ante story for the flipped direction:** large-margin accounts that are heavily net long are
  crowded, levered, funding-paying longs. A positive correlation with funding was visible in the
  first diagnostics (c_fund7 0.26-0.31), before the flip. Crowded longs underperform as they unwind.
- **Main risks:**
  - Only 2 real years of data.
  - The Dec-2021 fragment lost 10%.
  - Returns are lumpy (2024Q4).
  - Partly a funding effect.

### Candidate A7_S14: retail account crowding contrarian (pre-registered direction)
- **Rules:** score_t = -mean_14d(log global_ls). Other rules are identical to A9f.
- **IS (2021-12 to 2024-12):**
  - Headline: Sharpe 1.06, CAGR 22.0%, MaxDD -24.2%, t_NW **1.76**.
  - By year: 2022 +39.5% (Sharpe 1.62), 2023 +16.7% (1.02), 2024 +11.6% (0.59). It is decaying. Half-years 2023H2 -0.28 and 2024H1 0.13.
- **Stress tests:**
  - Cost and lag: 2x cost 0.91, 3x cost 0.75. Lag 1, 2 and 3 give 1.09, 1.01 and 0.88.
  - Plateau: S7 or S14 x rebalance 1/3/7 days give 0.59-1.13. Neighbour median about 0.9. Top-100: 1.28.
  - Concentration: leave-one-out minimum 0.87. Dropping the top-10 contributors leaves 0.65.
- **PnL source:**
  - Price vs funding: almost all price. Funding is +1.3%/yr and cost drag 3.2%/yr.
  - Long vs short leg: both legs add. Long-leg excess is +27.9%/yr (1.10); short-leg excess is +20.8%/yr (0.91).
- **Confounds:** correlation with 14d XS momentum is 0.33. The cross-sectional correlation of the score with 60d return is 0.43, because retail buys dips and shorts rallies. Momentum-neutral (resid on 60d return), it scores only **0.56** (2023 -0.30). It is not size (size-neutral 0.70) and not funding (correlation 0.01).
- **Diversification and size:**
  - Correlation with BTC -0.01.
  - Positions and capital: about 18 positions. Minimum equity about $200. $200 grew to $368.
- **Deflated Sharpe:** 0.45 at n=47, 0.38 at n=82.
- **Verdict:** weak. It meets the numeric bar on paper. But t is below 2, a third of the effect is momentum, and 2024 is fading.

### Blend (post-hoc): 50/50 of the A7_S14 and A9f weights. Both legs come from one hypothesis: crowded positioning is contrarian
- **IS:** Sharpe 1.52, CAGR 21.3%, MaxDD -11.1%, t_NW 2.63. By year: 2022 2.09, 2023 1.83, 2024 1.32.
- **Stress tests:** 2x cost 1.33, lag 1 1.58.
- **Correlations:** BTC -0.08, XS momentum 0.33.
- **Deflated Sharpe:** 0.75 at n=47, 0.69 at n=82.
- **Positions:** where the two books net, some positions become tiny. Drop any position under $10. In practice this needs about $400 or more for about 26 positions.
- This is the spec I would forward-test if the lead wants one book from family A. Freeze it exactly as written. Do not re-tune the smoothing.

## Family B: OI dynamics and taker flow. Result: none
Pre-registered grid: 21 configs, 8 robustness configs and 6 post-hoc configs.
- **B1 (-dOI over 1/3/7d):** Sharpe -1.03 to -1.42, turnover 260-570x a year.
- **B2 squeeze fuel, rank(dOI) - rank(ret):** -2.2 to -2.6. It is worse than plain reversal (B2c, -1.35 to -1.72).
- **Discrete quadrants:** -1.8 to -1.9. The price-down / OI-up quadrant does not bounce in dollar terms.
- **B3 leverage, OI/ADV30.** The level is -0.49. Of the dynamics signals, the best is -dlev7 (OI falling relative to volume means long) at 0.70. With a 3-day rebalance it scores 0.82, but 2x cost gives 0.33 / -0.12. Top-100 gives 0.41, q33 gives 0.31, fund-neutral 0.14, and 2023 is negative. **None.**
- **B4 taker imbalance (kline tbqv/qv, 2020-2024):** strongly negative in the contrarian direction in every year. The post-hoc continuation flip (+imb7) scores 1.16 (2021 2.15, 2022 1.12, 2023 -0.36, 2024 0.62). But its correlation with 14d XS momentum is **0.58**. Momentum-neutral it scores -0.21 (+imb3: -0.66). It is repackaged momentum. **None.**
- **B5 (metrics taker_ls_vol z30):** -3.85.
- **Repackaged funding?** The cross-sectional correlation of the B signals with 7d funding is small (|c| < 0.1, taker imbalance about 0.2). B3's return correlation with the funding factor is 0.02-0.08. They are not funding; they simply do not work.

## Survivorship exposure
The universe is the coins alive in 2026. A9f makes its money on the short leg, which this bias
understates, so it is conservative there. A7 earns from both legs, so its long side is somewhat
inflated. Neither result depends on one coin.

## Suggestions for the lead
1. Forward-test the blend, or A9f alone. Use the 2025+ forward test as the real test of the post-hoc sign. If A9f has a negative Sharpe in 2025, drop it.
2. A9f and h1's funding factor overlap (correlation 0.31-0.40 with fund7/fund30 L/S). If h1 forwards a funding factor, run the A9f-vs-funding spanning test again on the forward data.
3. Hyperliquid listing coverage of the top-50 Binance names was not checked. The signal needs Binance's top-trader and global ratio endpoints live, and they only give 30 days of history, which is enough for a 14-day mean.
4. Do not build anything on OI-change or taker-flow contrarian ideas in this universe. The fat right tail of the short leg kills them.

Scripts: `family_A*.py`, `family_B*.py`, `candidate_A7*.py`, `candidate_A9f*.py`, `final_summary.py`, `h2lib.py`.
Daily IS returns: `cand_A7_daily_IS.parquet`, `cand_A9f_daily_IS.parquet`.
