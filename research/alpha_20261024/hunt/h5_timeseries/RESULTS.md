# h5_timeseries — results (research data 2020-01-01 → 2024-12-31, loaded only through lab.py)

**Bottom line: none of my assigned families has an edge I would forward.** Seasonality (A) and BTC→alt
lead-lag (B1) are clean negatives. Volatility management (B2) gives no alpha over buy-and-hold. The one
lead I found, a post-hoc 4h "BTC shock → alt bounce" rule (B1R), is **weak**: Sharpe about 0.9, alpha
t-stat about 1.6, deflated-Sharpe probability 0.33, it fails the lag-1 test, and it overlaps the existing CLR
flush basket. I checked nothing on 2025+ data.

Costs: 0.07%/side plus real Binance funding in every number below. Configs run in total: **113**
(A 39, B1 28, B1R 18, B2 28).

## Family A — intraday / weekly seasonality (39 configs)
I picked windows and directions on 2020-22 (EST) and checked them on 2023-24 (VAL).
- **Day of week (a1_dow_1d.py, 9 configs).** Mon and Wed were positive in both halves for BTC and ETH
  (t about 1.2-1.7). The rank correlation of weekday means between EST and VAL was 0.46-0.57. This is
  just "be long on the drift days", and it loses to buy-and-hold:
  - BTC long Mon+Wed only: Sharpe 0.82 vs B&H 0.90. Lag-1 Sharpe is -0.39, and turnover costs 14.6%/yr.
  - Weekend-long and weekday-long both score below B&H. Weekends have about 40% lower vol but no
    different mean (weekend minus weekday t is about 0.1-1.1).
- **Hour of day (a2_hourly_1h.py, 24 configs).** The 24-hour return profile does not persist. The
  correlation between EST and VAL profiles is -0.06 (BTC), -0.23 (ETH) and -0.14 (EW20).
  - US-open hours 13-15 UTC were positive in 2020-22 and negative in 2023-24.
  - Gross edge is 0.3-4 bps per held hour, against a 14 bps round trip. Every window rule is deeply
    negative net: Sharpe -0.25 to -7.
  - Funding-conditional pre-settlement short (W8, |last 8h funding| ≥ 0.03%): BTC earns +2.6 bps
    price and +5.7 bps funding per trade, about 8 bps in total, below the 14 bps cost. Net Sharpe is
    -0.69 (BTC) to -1.78 (EW20).
- **Weekend-move reversal at the CME re-open (a3_weekend_reversal_1h.py, 6 configs).** I added this
  before looking at the conditional statistics. The weekend return has about 0 correlation with the
  next 24h return. Every config is negative (best: ETH, Sharpe -0.19). For BTC, if anything, there is
  weak continuation (t about 1.9), mostly in 2023-24. I did not flip the sign, because that would be
  post-hoc.
- **Verdict A: none.** The best attempt was BTC Mon+Wed long: Sharpe 0.82, below B&H, lag-1 -0.39.

## Family B1 — BTC → alt lead-lag (28 configs, b1_leadlag.py)
I regressed the alt basket's next-bar return on BTC's last bar, controlling for the alts' own last bar,
using Newey-West (HAC) t-stats.
- **The sign is the opposite of the hypothesis.** The BTC coefficient is -0.074 (t -2.4) at 4h and
  -0.066 (t -3.4) at 1h. Same sign in every sub-period, significant only in some.
- BTC also reverses its own last bar: -0.05 to -0.06, t about -2 to -3.
- There is no "alts catch up to BTC" effect at 1h or 4h; HFT arbitrage presumably removes it at
  sub-minute speed.
- Directional, hedged and cross-sectional catch-up rules all lose. The best was 4h directional
  k=2 hold 3: Sharpe -0.10.
- The 1h cross-sectional beta-residual reversal has **gross** Sharpe 4.6, but turnover is 12,700x a
  year and costs are 887%/yr. That is microstructure, which you can only harvest as a maker.
- **Verdict B1: none.**

## Family B1R — post-hoc: 4h BTC shock reversal (18 configs; the deflated Sharpe uses 46 trials)
Rule: go long an equal-weight top-20 alt basket (BTC excluded) for h 4h-bars after BTC's 4h return
z-score drops below -k.
- **Story:** liquidation and stop cascades overshoot, and the bounce pays whoever supplies liquidity.
- **Plateau:** Sharpe is positive in all 18 configs (0.15-0.94, median about 0.7). Hold-1 versions
  are 0.72-0.91.
- **Best clean config, `down k2.5 h1`:**
  - Performance: Sharpe 0.91 (gross 1.10), CAGR 19.8%, MaxDD -23.6%, t_NW 2.05.
  - By sub-period: 2020-21 1.03, 2022 0.24, 2023-24 1.27. Calendar 2020 was -13%.
  - Stress tests: 2x cost 0.72, **lag-1 -0.81** (the bounce is over within 4h).
  - Market exposure: corr BTC 0.16, corr 14d XS-momentum 0.00. Alpha vs alt B&H is +18%/yr
    (t 1.62, beta 0.04).
  - Trading: 164 events, in the market 1.5% of the time. Price gives +24.7%/yr, funding +0.1%/yr,
    costs -4.5%/yr.
  - Size: about 18 coins per event, so it needs at least $190 of equity at 1x for $10 minimum orders.
    Starting from $200, IS equity ends at $477, with a low of $152.
- **Lag-robust neighbour, `down k1.5 h6`:**
  - Sharpe 0.94, 2x cost 0.75, lag-1 0.57. Sub-periods: 1.59 / 0.17 / 0.55.
  - **But 2021 alone made +390%**, and correlation with BTC is 0.51. This one is mostly alt-season beta.
- **Diagnostics (b1r_diag.py):**
  - The top 10% of held bars produce more than 100% of gross PnL.
  - Sharpe stays positive without Mar-2020 and May-2021.
  - The bounce is spread over hours 1-4 after the trigger, so a few minutes of execution delay is fine,
    but a full 4h delay is not.
- **Concerns:**
  - The rule came from looking at the data.
  - It does not reach Sharpe 1 and fails lag-1.
  - It is heavily exposed to **survivorship**: it buys alt dips, and the coins whose dips never bounced
    (LUNA, FTT and others) are not in the universe.
  - It is close to CLR's liquidation-flush long.
- **Verdict: weak.** Use it as a control or cross-check for CLR, not as a new strategy.

## Family B2 — volatility-managed exposure (28 configs, b2_volmanaged.py)
Exposure is scaled as min(cap, 50% / trailing vol) on BTC or an equal-weight top-20 basket, at 1d and
4h, with optional drawdown control.
- **Vol scaling adds nothing.** Alpha vs same-asset B&H has |t| < 1.1 everywhere, and Sharpe moves by
  about ±0.1 around B&H. Every config is negative in 2022.
- For EW20, vol scaling cuts the 2021 alt-season up-vol, and 4h scaling is worse.
- The best attempt, BTC with a 30d vol target and flat when more than 20% below the 180d high, reaches
  Sharpe 1.11. But it is a trend/drawdown filter, which is RNT territory: it stayed flat in 2022 and
  made -18% in 2021. Neighbours were 0.96 (DD30) and 0.80 (DD40). The EW20 drawdown versions are below B&H.
- Side fact: holding long perps cost about **10-12%/yr in funding** over 2020-24, which matters for
  every long-only perp bot.
- **Verdict B2: none.**

## Suggestions worth a separate test (not tested here)
1. **Funding-settlement capture with maker execution.** Price plus funding is about 8 bps per trade
   against a 14 bps taker round trip. A maker-only round trip of about 4 bps might clear it. This
   needs a fill-probability or order-book simulation first.
2. **Use B1R as an entry-timing filter for CLR or other dip-buy logic, not as its own strategy.** Enter
   at the close of the 4h bar that triggered the signal, and expect the bounce to be gone within 4h.
3. **Do not time trades by hour of day or day of week.** These profiles flip sign between 2020-22 and
   2023-24.

Files (all in this folder): common.py, a1_dow_1d.py, a2_hourly_1h.py (+log), a3_weekend_reversal_1h.py (+log),
b1_leadlag.py (+log), b1r_shock_reversal_4h.py (+log), b1r_diag.py (+log), b2_volmanaged.py, final_eval.py (+log),
and a *_results.csv file for every grid.
