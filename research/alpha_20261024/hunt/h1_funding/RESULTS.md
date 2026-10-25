# h1_funding: results (research period 2020-01 to 2024-12 only, via lab.py)

All numbers are net of 0.07%/side perp cost plus actual Binance funding, on 1d bars, unless stated otherwise.
The 2025+ data was never loaded.
Scripts are in this folder: `famA_grid.py`, `famA_deep.py`, `famA_conc.py`, `famA_size.py`, `famA_checks.py`, `famA_shortleg.py`,
`famB_carry.py`, `famB_settle.py`, `final_numbers.py`, `dsr_calc.py`, plus the helper `common.py`. CSV/JSON outputs sit beside them.

Configs tried: **Family A = 52**. That is 42 pre-registered, plus 10 post-hoc variants that are labelled as such.
**Family B(i) = 19**, all pre-registered. **B(ii) = 0 rules**: it is an event study, and no rule passed the gate. Total: 71.

## Family A: cross-sectional funding factor. Verdict: PROMISING

**Rationale.** High trailing funding marks a crowded leveraged long. The long side pays a carry premium to whoever takes the other side.
When that crowding unwinds, the price falls too.
Who is on the other side: retail leveraged longs who chase hot alts and are price-insensitive to an 8-hourly fee.
Arbitrage capital that would short these names is limited, because there is no spot borrow on alts and squeeze risk is high.
That limit is why the premium persists.

**Grid (pre-registered, 42 configs).**
- A1: score = -(sum of funding over the last L days), L in {1, 3, 7, 14}. Universe top-50 or top-100. Weighting quintile or rank. Rebalance every 1 or 3 days. 32 configs.
- A2: change in funding over 3 or 7 days. 8 configs.
- A3: funding residualised on 14d momentum. 2 configs.
- "Relative to BTC funding" subtracts the same number from every coin on a given day, so its cross-sectional rank is identical to A1. It was not run separately.
- Post-hoc (10 configs): tercile and top-30 implementations (A4), funding residualised on liquidity (A5), and a short-leg-only version with hedges (A6).

**Main findings.**
- Short windows are noisy. L1 and L3 with quintile weighting give Sharpe 0.4 to 0.8 and turn negative in 2022-24.
- **Persistent funding works.** With L7 or L14 and rank weighting, Sharpe is 1.47 to 1.71 across all 8 universe/rebalance combinations, a plateau rather than a spike. Quintile versions give 0.99 to 1.44.
- Delta-funding (A2) has no edge: Sharpe is -0.6 to 0.8, and 2022 and 2023-24 are negative.
- **Price vs funding.** Funding carry is about +10 to 11% a year, almost all of it received on the short leg (+11% short, -1.5% long). The price part alone, net of costs, has Sharpe about 1.0 to 1.3. So this is **not pure carry**.
  - The price PnL measured on spot returns (0.30/yr, Sharpe 1.56) matches the PnL on perp returns (0.31/yr, Sharpe 1.60). So it is real spot underperformance, not basis convergence.
- **The legs are uneven.** Against an equal-weight universe benchmark, the short leg has Sharpe 2.2 (positive in every year, 1.1 to 4.0). The long leg has Sharpe 0.6, and it is about 0 in 2020, 2023 and 2024.
  - The edge is "short the crowded longs".
  - Survivorship bias, since only coins alive in 2026 are included, understates the short leg and inflates the long leg. The real short edge is probably larger and the long edge smaller.
- **It is not a known factor.** Correlation with 14d XS momentum is +0.09, with BTC -0.07, and with 3d reversal +0.06.
  - Residualising the signal on momentum keeps Sharpe 1.4. Residualising on liquidity keeps Sharpe 1.6 to 1.8.
  - The two legs have about equal liquidity rank (0.53 vs 0.46).

**Best candidate: A4_L14_top50_tercile_rb1** (an implementable version of the pre-registered A1_L14 rank signal).
- **Rules.** Each day at 00:00 UTC, take the point-in-time top-50 perps by 30d median quote volume, with at least 60 days of listing. Compute F14 = the sum of the last 14 days of funding settlements.
  - Short the top third by F14, equal weight, -0.5 gross in total.
  - Long the bottom third, equal weight, +0.5 gross in total.
  - Rebalance daily. There are no stops; positions change only through rebalancing.
- **IS 2020-03 to 2024-12:** CAGR 37.5%, Sharpe 1.61, t_NW 3.37, MaxDD -28.9%, vol 21%, skew +2.8.
  - Turnover 64x a year, cost drag 4.5% a year, funding +9.8% a year, about 26 positions.
- **Yearly returns:** 2020 +18.5%, 2021 +149%, 2022 +23.3%, 2023 +9.8%, 2024 +16.8%. Yearly Sharpe: 1.02, 2.89, 1.78, 0.79, 0.99.
- **Sub-periods:** 2020-21 2.18, 2022 1.78, 2023-24 0.90.
- **Stress tests:** 2x cost gives Sharpe 1.40. Lag 1 gives 1.65. Median Sharpe of the neighbouring configs is about 1.46.
- **Deflated Sharpe probability:** 0.87 with n=52, 0.84 with n=71.
- **Pre-registered twin, A1_L14_top50_rank_rb1:** Sharpe 1.53, 2x cost 1.31, lag 1 1.52, sub-periods 2.19 / 1.72 / 0.64.
- **Min equity for $10 per position:** about $340 at 1x gross. At $200 that needs about 1.4x gross.
- **$200 starting capital:** grows to $237 at end-2020, $591 at end-2021, $729 at end-2022, $801 at end-2023 and $935 at end-2024.

**Concerns.**
1. **2021 dominates.** It returned +149%, and more than half of the total log-return came from the alt mania.
2. **There are long weak stretches.** The rolling 365-day Sharpe bottomed at -1.45 in June 2024, so a year-long losing stretch is possible.
3. **PnL is concentrated.** The top 5 coins (CRV, BNB, LINK, TRX, FIL, mostly long-side large caps) produce 53% of the PnL.
   - Dropping them with hindsight leaves Sharpe 1.21, but 2023-24 falls to 0.69.
4. **The long leg is close to worthless after 2022.** The edge comes from the short leg.
5. **This is Binance funding.** On HYPE, funding is hourly with different levels and a different set of coins. The signal must be rebuilt from HYPE funding before trading there.

**Variant worth a look: A6b (post-hoc).** Short the top-funding third and hedge with an equal-weight long of the whole top-50.
- Sharpe 1.66, CAGR 21%, MaxDD -15%. 2x cost gives 1.44, lag 1 gives 1.73. Sub-periods 2.17 / 2.00 / 0.97.
- It needs about 39 positions, so the minimum equity is about $1,000.
- Hedging the short leg with BTC+ETH instead (A6a) fails, with Sharpe 0.46, because of the 2021 alt-season.

## Family B(i): delta-neutral cash-and-carry. Verdict: WEAK as an alpha, useful as a yield sleeve

**Setup.** Long spot and short perp. PnL = spot return - perp return + funding. Cost is 0.10% + 0.07% per side on changes in notional.

**Grid (19 configs).**
- Threshold on trailing 3d or 7d funding (0, 10, 20 or 40% APR), on top-50 or on BTC+ETH only. 16 configs.
- Always-on BTC+ETH. 1 config.
- Top-N (3 or 10) by funding. 2 configs.

**Results.**
- **Selecting high-funding coins destroys value.** Funding mean-reverts after entry, and each switch costs 0.34% round trip.
  - Selective alt versions earn -4% to +8% APR and lose in 2022 (down to -29%).
  - Top-3 by funding earns -0.4% APR.
- **The best configuration is the dumbest one: always-on BTC+ETH.**
  - 16.7% CAGR on notional, about 12.3% on capital with the perp at 3x margin. Sharpe 9.5, MaxDD -1.7%.
  - Yearly: 2020 +22.8%, 2021 +40.4%, 2022 +2.4%, 2023 +8.3%, 2024 +13.3%.
  - 2021 supplied 44% of the log-return. Funding is +15.8% a year; basis is -0.3% a year.
  - 2x cost gives Sharpe 9.4; lag 1 gives 9.5.
- **Why it is only WEAK.** This is a market-wide carry premium, not selection alpha. Its level depends on the leverage regime: 2022 returned only +2.4%.
  - The Sharpe of 9 understates tail risk: exchange or custody failure (FTX), USDT depeg, and perp margin calls in sharp rallies when spot and perp sit on separate accounts.
  - Minimum equity is about $30.

## Family B(ii): funding-settlement timing on 1h bars. Verdict: NONE

**Setup.** About 215k coin-settlement events in the top-50. Returns are cross-sectionally demeaned in windows around each settlement.

**Results.**
- **The only large effect cannot be traded.** When the realised funding at the settlement is ≤ -0.05%, the coin returns +45bp over the 3 hours before settlement (t 7.6). But that rate is only fully known at settlement, and the price move and the funding are produced together by the same squeeze.
  - Conditioning on the previous settlement's funding, which is known in advance, the effect falls to +2.6bp (t -0.4).
- **The "high funding dips into settlement and then recovers" pattern does not exist.** For funding ≥ 0.1%: 3h before is -8bp (t -0.9) and 1h after is +0.4bp.
- Every after-settlement window is under 10bp in absolute size, far below the 30bp gate. No rule was built.

## Recommendation to lead

Forward-test the A4_L14_top50_tercile_rb1 spec.
- If you prefer the purely pre-registered version, use A1_L14_top50_rank_rb1.
- Report A6b alongside it as a lower-volatility variant.

Expect forward Sharpe nearer the 2023-24 level (about 0.9) than the full-sample 1.6.
