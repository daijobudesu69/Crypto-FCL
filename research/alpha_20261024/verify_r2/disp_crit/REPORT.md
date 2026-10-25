# disp_crit — overfitting / survivorship / risk critique of DISP7 (dispersion-timed alts-vs-BTC spread)

Verifier: disp_crit (independent). Data: lab.py only, IS 2020-01-01..2024-12-31, ALPHA_FORWARD never set,
no 2025+ data read. Signal inputs: close (+ qv through lab.hype_universe). Cost lab.HYPE_COST + Binance funding.
Own re-implementation in `dc.py` (does not import the hunter's code). Scripts `s0_repro.py` .. `s6_*.py`.
Results are appended step by step.

## Step 0 — reproduction
Independent implementation: Sharpe 1.38, CAGR +37.6%, MaxDD -21.1%, t_NW 2.90,
skew 0.16, excess kurt 5.40, days 1646 (start 2020-06-30), turnover/yr 39,
cost/yr 2.5%, funding/yr -1.3%, avg n_pos 36.7.
Sub-periods 20-21 1.70, 22 1.31, 23-24 1.11; 2x cost 1.28; lag1 1.43.
Switches 60; days in LONG-SPREAD 846 (ann +48.9%, SR 1.64), SHORT-SPREAD 800 (ann +20.8%, SR 1.02).
Yearly: 2020: -1.6% (SR 0.03); 2021: +115.3% (SR 2.44); 2022: +30.1% (SR 1.31); 2023: +63.4% (SR 2.87); 2024: -6.2% (SR -0.14).
Claimed: 1.38 / -21% / 1.70, 1.31, 1.11 / 2x 1.28 / lag1 1.43 / ~60 switches -> REPRODUCED.

## Step 1 — multiple testing
SR 1.375, T 1646 d (4.51 y), skew 0.16, raw kurt 8.40. t (iid) 2.92, t_NW(10) 2.90.

| trials n | DSR (lab default trial-SR sd 0.5) | DSR (trial-SR sd = family-B empirical 0.45) | E[max SR] null (sd 0.5) |
|---|---|---|---|
| 12 | 0.876 | 0.908 | 0.83 |
| 53 | 0.684 | 0.764 | 1.15 |
| 95 | 0.600 | 0.696 | 1.26 |
| 855 | 0.313 | 0.438 | 1.61 |
(B1 spread signal family, 12 pre-registered: trial-SR sd 0.49; family B all 53: sd 0.45.)

Harvey-Liu-Zhu: t_NW = 2.90 < 3.0 -> FAILS the t>3 hurdle. Bonferroni haircut (two-sided p single = 0.0038):
- n=12: adj p 0.045 -> haircut t 2.00 -> haircut Sharpe 0.94 (31% haircut)
- n=53: adj p 0.199 -> haircut t 1.28 -> haircut Sharpe 0.60 (56% haircut)
- n=95: adj p 0.357 -> haircut t 0.92 -> haircut Sharpe 0.43 (68% haircut)
- n=855: adj p 1.000 -> haircut t 0.00 -> haircut Sharpe 0.00 (100% haircut)

20-day circular block bootstrap (10,000): median SR 1.38, 90% CI [0.59, 2.18], P(SR<0.5) = 0.034, P(SR<0) = 0.002, P(SR<1.0) = 0.212.
60-day blocks (regime spells avg 27d, so 20d blocks understate dependence): 90% CI [0.68, 2.04], P(SR<0.5) = 0.017.
Circular-shift timing null (227 rotations of the state path, same spell structure and same time-in-state): null SR mean -0.14, sd 0.52, 95th pct 0.58, max 1.35; P(null >= 1.38) = 0.000.
Unconditional spread over the same dates: always LONG-SPREAD SR 0.56, always SHORT-SPREAD SR -0.62.

## Step 2 — stability
Rolling 365d Sharpe (1282 obs): min -0.27 (2024-12-22), max 2.97 (2024-01-04), median 1.44; share of windows > 0: 99%, > 0.5: 94%, > 1.0: 70%.
Quarter-end values: 2021-06 2.44, 2021-09 2.36, 2021-12 2.44, 2022-03 1.40, 2022-06 1.05, 2022-09 1.00, 2022-12 1.31, 2023-03 1.67, 2023-06 1.62, 2023-09 1.83, 2023-12 2.87, 2024-03 2.06, 2024-06 0.62, 2024-09 0.84, 2024-12 -0.17
Leave-one-year-out Sharpe: ex2020 1.57, ex2021 0.96, ex2022 1.40, ex2023 1.12, ex2024 1.78
ex-2021: 0.96 (CAGR +21.2%); 2023-24 alone 1.11; 2024 alone -0.14; 2022-24 1.18; 2020-H2 0.03; last 18 months (2023-07..2024-12) 0.47.
Ex 10 best days: 0.93; ex 20 best 0.57; ex 10 best & 10 worst 1.54. 10 best days sum +57.5% vs total arithmetic +158.9%; dates: 2021-01-16, 2020-11-23, 2021-02-10, 2021-01-09, 2023-06-10, 2021-06-21, 2020-11-21, 2021-01-28, 2022-05-11, 2021-05-24
Regime episodes (contiguous state spells): 61 total (30 LONG-SPREAD, 31 SHORT-SPREAD); median length 14 d, mean 27 d, longest 177 d. Episodes with P&L > 0: 56%. t-stat of mean episode log-P&L across episodes: 2.03 (n=61).
  state +1: 30 episodes, win 60%, mean +3.4%, median +0.7%, t 1.69
  state -1: 31 episodes, win 52%, mean +1.3%, median +0.7%, t 1.13
Total log P&L +1.44. Top-3 episodes = 69% of it, top-5 = 85%, top-10 = 109%.
Top-5 episodes: 2020-11-04..2021-04-29 st +1 177d +58.5%; 2023-02-25..2023-06-10 st -1 106d +24.3%; 2021-12-02..2022-05-11 st -1 161d +16.1%; 2020-10-08..2020-11-03 st -1 27d +13.3%; 2021-05-05..2021-05-31 st +1 27d +9.9%
Worst-5 episodes: 2020-09-25..2020-09-30 st -1 6d -4.5%; 2023-07-07..2023-07-13 st -1 7d -4.5%; 2024-02-05..2024-06-06 st +1 123d -4.9%; 2022-06-29..2022-07-18 st -1 20d -5.5%; 2021-07-17..2021-08-18 st -1 33d -7.7%
Leave-one-episode-out Sharpe: min 1.02, median 1.38, max 1.51.
  top-1 episodes removed (days dropped): Sharpe 1.02; set flat instead: 0.96
  top-3 episodes removed (days dropped): Sharpe 0.69; set flat instead: 0.59
  top-5 episodes removed (days dropped): Sharpe 0.43; set flat instead: 0.36
  top-10 episodes removed (days dropped): Sharpe -0.11; set flat instead: -0.09

## Step 3 — survivorship
L14 = ETH, ZEC, XRP, BNB, ADA, LINK, BCH, LTC, DASH, XLM, XMR, ETC, TRX, ATOM; L8 = ETH, XRP, BCH, LTC, BNB, ADA, LINK, TRX. Coins in HYPE top-50 that were listed before 2022: 34 of 69 ever-members.

| variant (signal universe -> traded universe) | Sharpe | 20-21 / 22 / 23-24 | 2024 | LONG-SPREAD ann (SR) | SHORT-SPREAD ann (SR) | alt-leg ann | BTC-leg ann | state agree w/ base | avg names |
|---|---|---|---|---|---|---|---|---|---|
| base U50 -> U50 | 1.38 | 1.70 / 1.31 / 1.11 | -0.14 | +49% (1.64) | +21% (1.02) | +41% | -5% | 1.00 | 37 |
| (a) U50 -> pre-2022 listings | 1.43 | 1.70 / 1.28 / 1.27 | 0.20 | +48% (1.62) | +23% (1.19) | +41% | -5% | 1.00 | 30 |
| (a) pre-2022 -> pre-2022 | 1.55 | 1.70 / 1.25 / 1.64 | 0.98 | +55% (1.85) | +24% (1.23) | +42% | -3% | 0.92 | 30 |
| (b) U50 -> L14 majors | 0.84 | 1.22 / 0.34 / 0.65 | 0.46 | +26% (0.90) | +14% (0.78) | +25% | -5% | 1.00 | 15 |
| (b) L14 -> L14 | 0.58 | 1.52 / -1.47 / 0.39 | 0.44 | +23% (0.75) | +7% (0.38) | +6% | +8% | 0.74 | 15 |
| (b) U50 -> L8 majors | 0.94 | 1.48 / 0.54 / 0.54 | 0.42 | +34% (1.20) | +10% (0.58) | +27% | -5% | 1.00 | 9 |
| (b) L8 -> L8 | 0.52 | 0.87 / -0.45 / 0.58 | 1.01 | +27% (0.93) | -0% (-0.02) | +9% | +4% | 0.73 | 9 |
| (c) U50 -> ex first-180d | 1.18 | 1.33 / 1.26 / 1.04 | -0.15 | +38% (1.24) | +23% (1.15) | +36% | -5% | 1.00 | 33 |
| (c) ex-180d -> ex-180d | 1.36 | 2.34 / 1.44 / 0.44 | -0.91 | +50% (1.57) | +24% (1.16) | +25% | +11% | 0.88 | 35 |
Base alt leg in state +1: total arithmetic P&L +220.0%; from coins in their first 180d +74.2% (weight share 16%), first 365d +141.1%.
Base alt leg in state -1: total arithmetic P&L -6.7%; from coins in their first 180d -9.9% (weight share 7%), first 365d -24.4%.
  state +1: P&L from coins listed 2022+ +27.1%
  state -1: P&L from coins listed 2022+ -8.7%
Top-8 alt contributors in LONG-SPREAD state: DOGE +16.4%, SOL +10.8%, SUSHI +10.2%, BNB +9.9%, CRV +8.4%, AVAX +8.1%, XRP +7.9%, ATOM +7.8% | sum all +220.0%, ex top-5 +164.4%
Worst-5 alt contributors in SHORT-SPREAD state (short survivors that rallied): 1000SHIB -3.9%, ETH -2.6%, DOGE -2.5%, ETC -2.1%, INJ -1.9% | best-5: ZEN +3.1%, ALGO +3.1%, FIL +2.9%, HBAR +2.3%, GALA +2.0% | sum -6.7%

## Step 4 — plateau (19 pre-registered cells, written in s4_plateau.py header before running)

| cell | Sharpe | 20-21 | 22 | 23-24 | 2024 | 2x cost | switches |
|---|---|---|---|---|---|---|---|
| L5_w180 | 1.96 | 2.18 | 1.27 | 2.14 | 1.84 | 1.80 | 107 |
| L5_w365 | 1.33 | 1.79 | 1.31 | 0.88 | 0.05 | 1.20 | 82 |
| L5_w730 | 1.36 | 1.78 | 0.79 | 1.26 | 0.31 | 1.24 | 78 |
| L7_w180 | 1.91 | 1.56 | 2.13 | 2.32 | 1.59 | 1.78 | 89 |
| L7_w365 | 1.38 | 1.70 | 1.31 | 1.11 | -0.14 | 1.28 | 60 |
| L7_w730 | 1.26 | 1.40 | 0.91 | 1.33 | 0.12 | 1.16 | 56 |
| L10_w180 | 1.52 | 1.16 | 1.35 | 2.12 | 1.56 | 1.41 | 69 |
| L10_w365 | 1.70 | 1.68 | 1.65 | 1.84 | 0.99 | 1.62 | 45 |
| L10_w730 | 1.10 | 0.86 | 1.82 | 1.06 | -0.23 | 1.02 | 43 |
| L14_w180 | 1.16 | 0.71 | 1.58 | 1.55 | 0.37 | 1.08 | 41 |
| L14_w365 | 1.70 | 1.84 | 1.59 | 1.70 | 0.97 | 1.61 | 50 |
| L14_w730 | 1.35 | 1.81 | 0.90 | 1.14 | -0.07 | 1.28 | 36 |
| band0.60/0.40 | 1.35 | 1.43 | 1.17 | 1.42 | 0.40 | 1.22 | 84 |
| band0.75/0.25 | 1.48 | 1.83 | 1.62 | 1.10 | -0.34 | 1.40 | 48 |
| band0.80/0.20 | 1.41 | 1.58 | 1.65 | 1.18 | -0.16 | 1.34 | 33 |
| top30 | 1.30 | 1.59 | 1.19 | 1.10 | -0.13 | 1.18 | 80 |
| top75 | 1.39 | 1.70 | 1.31 | 1.15 | -0.03 | 1.29 | 60 |
| est_iqr | 1.09 | 1.56 | 1.46 | 0.41 | -0.70 | 0.92 | 124 |
| est_mad | 1.11 | 1.52 | 1.93 | 0.27 | -1.12 | 0.94 | 116 |
| **MEDIAN** | **1.36** | 1.59 | 1.35 | 1.18 | 0.05 | 1.28 | 60 |
Spread: min 1.09, IQR [1.28, 1.50], max 1.96; cells > 1.0: 19/19, > 0.5: 19/19; cells with all 3 sub-periods > 0: 19/19; cells with 2024 > 0: 10/19. Baseline rank among 19: 9.

## Step 5 — risk
Worst day -10.6% (2021-05-19, state +1, BTC -14.4%, alt EW -36.0%); best day +7.6% (2021-05-24). Worst calendar week -12.4% (w/e 2021-05-23), worst rolling 7d -12.5% (to 2020-12-23); worst calendar month -15.8% (2020-12), worst rolling 30d -19.2% (to 2021-01-02). Daily vol 26%/yr; 1st pct day -3.1%; months negative 33%.
Worst 5 days: 2021-05-19 -10.6% (st +1, BTC -14.4%, altEW -36.0%); 2020-12-23 -6.0% (st +1, BTC -2.5%, altEW -14.5%); 2024-12-09 -5.6% (st +1, BTC -3.8%, altEW -15.0%); 2021-02-08 -5.6% (st +1, BTC +19.6%, altEW +8.5%); 2024-04-12 -5.3% (st +1, BTC -4.2%, altEW -14.7%)
Worst drawdown -21.1%: peak 2020-11-24 -> trough 2021-01-02 (39 d), recovered 2021-01-23. During it: alt leg +3.3%, BTC leg -26.2% (arith.), days in LONG-SPREAD 39, SHORT-SPREAD 0, switches 0; BTC +67.9%, alt EW +3.4%.
All drawdowns deeper than -10%: 2020-11-25->2021-01-02 -21.1% (rec 2021-01-23); 2024-01-02->2024-11-20 -18.0% (rec 2024-12-06); 2021-06-23->2021-08-26 -17.9% (rec 2022-04-30); 2020-08-17->2020-10-13 -16.0% (rec 2020-11-21); 2021-05-19->2021-05-23 -15.2% (rec 2021-06-11); 2022-06-26->2022-08-02 -14.7% (rec 2023-03-04); 2024-12-07->2024-12-21 -14.1% (rec open); 2021-02-14->2021-03-14 -10.3% (rec 2021-04-04)
Drawdown at 2024-12-31: -8.1%; equity high set on 2024-12-06.

| event (return dates) | state(s) | strategy | BTC | alt EW | alt leg | BTC leg |
|---|---|---|---|---|---|---|
| 2021-01-21 dip | +1 | +3.0% | -8.1% | -2.7% | -0.7% | +3.7% |
| 2021-05-12..23 China/Elon crash | +1 | -6.2% | -38.9% | -51.1% | -28.4% | +23.1% |
| 2021-05-19 alone | +1 | -10.6% | -14.4% | -36.0% | -17.8% | +7.2% |
| 2021-12-04 flash crash | -1 | +3.7% | -12.6% | -19.2% | +10.2% | -6.5% |
| 2022-05-05..13 LUNA | -1/+1 | +11.0% | -26.2% | -39.4% | +26.2% | -15.3% |
| 2022-06-10..19 3AC/Celsius | +1 | +4.9% | -31.7% | -24.5% | -12.5% | +17.6% |
| 2022-11-06..12 FTX | +1 | -3.3% | -21.1% | -28.5% | -13.7% | +10.5% |
| 2023-03-09..14 SVB | -1 | +0.9% | +13.8% | +11.6% | -5.9% | +6.9% |
| 2023-06-05..12 SEC suits | -1/+1 | +9.4% | -4.5% | -21.3% | +11.6% | -2.4% |
| 2024-03-11..20 alt top | +1 | -2.2% | -1.6% | -5.9% | -2.8% | +0.6% |
| 2024-04-12..14 Iran | +1 | -5.4% | -6.3% | -17.4% | -8.5% | +3.2% |
| 2024-08-02..06 yen carry | -1/+1 | -1.3% | -14.3% | -19.7% | -2.7% | +1.5% |
| 2024-12-09..10 alt flush | +1 | -6.2% | -4.5% | -16.7% | -8.5% | +2.3% |

State +1: BTC beta +0.06 (corr +0.13), beta to alt EW +0.23; on BTC < -3% days (n=107) mean strat -0.62%, on BTC > +3% days (n=129) +0.03%; on alt-EW < -5% days (n=95) mean strat -1.71%.
State -1: BTC beta -0.06 (corr -0.16), beta to alt EW -0.18; on BTC < -3% days (n=79) mean strat +0.58%, on BTC > +3% days (n=99) -0.01%; on alt-EW < -5% days (n=65) mean strat +1.54%.
Trailing reality check: full-sample beta of alt EW to BTC 1.12 -> the 0.5/0.5 spread is NOT beta-neutral: LONG-SPREAD carries net long beta, SHORT-SPREAD net short beta. Full-sample corr strat vs BTC +0.03.

Small-account replay (account_sim, $10 min order, HYPE cost, signal from U50, traded on HYPE top-11 = 10 alts + BTC):

| start equity | end equity | total ret | MaxDD | orders | skipped < $10 | fees $ | funding $ | avg n_pos |
|---|---|---|---|---|---|---|---|---|
| $200 | $302 | +51% | -45.1% | 1385 | 3270 | 27 | +9 | 9.0 |
|  (SR 0.47; 2024 -23%; 2023-24 SR 0.16) | | | | | | | | |
| $300 | $731 | +144% | -26.9% | 1906 | 0 | 68 | -46 | 11.0 |
|  (SR 0.89; 2024 -22%; 2023-24 SR 0.13) | | | | | | | | |
lab.backtest of the same 11-pos weights: SR 0.91, subs 20-21 1.06, 22 1.92, 23-24 0.21; 2020: -10.3% (SR -0.54); 2021: +75.0% (SR 1.80); 2022: +51.6% (SR 1.92); 2023: +29.9% (SR 1.55); 2024: -19.6% (SR -0.84).

## Step 6 — diagnostics for the rationale critique
corr(disp7 percentile, trailing 7d alt-BTC spread) +0.29; with trailing 30d spread +0.31; with 7d BTC return +0.13; with xs median 7d return +0.17.
Upside vs downside dispersion: corr(p, 90th-median) +0.65, corr(p, median-10th) +0.49; mean (90th-median)/(median-10th) when LONG-SPREAD 1.84 vs SHORT-SPREAD 1.40.
Universe growth: corr(p, # members in first 180d) +0.36; mean young members LONG-SPREAD 4.8 vs SHORT-SPREAD 2.6; corr(p, universe size) -0.12.

| year | % days LONG-SPREAD | LONG-SPREAD ann ret (SR) | SHORT-SPREAD ann ret (SR) | alt EW - BTC (log, yr) |
|---|---|---|---|---|
| 2020 | 79% | -17% (-0.56) | +70% (3.64) | -0.59 |
| 2021 | 60% | +154% (4.09) | -25% (-0.93) | +1.80 |
| 2022 | 17% | +59% (1.93) | +23% (1.14) | -0.43 |
| 2023 | 43% | +61% (3.23) | +43% (2.57) | -0.06 |
| 2024 | 72% | -11% (-0.43) | +17% (0.86) | -0.42 |
Mean fwd 7d alt-BTC log spread (non-overlapping weekly samples), state +1 minus state -1, by year: 2020 +3.96%, 2021 +2.65%, 2022 +3.23%, 2023 +3.32%, 2024 +0.34%

### Step 6 — economic rationale critique
Claimed story: high cross-sectional dispersion = capital rotating into alt stories, so alts beat BTC; low dispersion =
macro/deleveraging, BTC-led tape, so alts lag. What the diagnostics say:
- High disp7 is mostly **right-tail dispersion (pumps)**: corr(p, 90th pct minus median) +0.65, corr(p, median minus 10th) +0.49;
  the up/down tail ratio is 1.84 in LONG-SPREAD vs 1.40 in SHORT-SPREAD. It overlaps with trailing spread momentum
  (corr +0.29 / +0.31 with the 7d / 30d alt-BTC spread). Still, the circular-shift null (P < 0.005) and the hunter's alpha t of 2.9
  against spread TS-mom suggest the timing is more than plain spread momentum in-sample.
- **It is partly a new-listing / survivor effect.** p correlates +0.36 with the number of top-50 members in their first 180 days
  (4.8 young members in LONG-SPREAD vs 2.6 in SHORT-SPREAD). Young survivors are exactly the coins that pumped *and* survived to 2026.
  So the same survivorship lifts the signal and the long-alt payoff together.
- A plausible persistent mechanism exists: alt-season reflexivity (retail and narrative flows chase dispersion-generating winners,
  slow capital follows for weeks), plus deleveraging regimes where alts lose to BTC. This is a slow (27-day spells), capacity-unconstrained
  market-level timing signal. No single-name arbitrage removes it, so it could persist.
- **When it stops working:** when dispersion comes from idiosyncratic token events (unlocks, hacks, ETF-ticker rotation, memecoin
  launches) instead of broad alt rotation, and when flows are BTC-specific (spot-ETF / treasury-company buying). 2024 is that regime:
  72% of days in LONG-SPREAD, which lost 11%/yr. The state +1 minus state -1 forward 7d spread gap fell to +0.34% (vs +2.7..4.0% in 2020-23).
  There was an 11-month drawdown (2024-01-02 -> 2024-11-20, -18%). The worst drawdown overall (-21%, Nov-Dec 2020) is the same
  pattern: a BTC-led rally (BTC +68%, alts +3%) while the book was short BTC. Rolling 365d Sharpe ends 2024 at -0.17, its sample low.
- The book is $-neutral but not beta-neutral (alt beta to BTC 1.12). LONG-SPREAD loses -1.7% on average on days when alts fall more than 5%
  (2021-05-19: -10.6%; 2024-04-13: -5.3%; 2024-12-09: -5.6%). An alt flash crash while in LONG-SPREAD is the main tail risk. The state
  is long-spread about half the time.

## Verdict (disp_crit)
**confirmed_with_concerns.** The headline numbers reproduce exactly (SR 1.38, MaxDD -21%, 60 switches). The parameter surface is a genuine
plateau (19 pre-registered cells: median 1.36, min 1.09, all > 1.0). Timing beats 227 circular-shifted state paths (max null 1.35).
But:
1. Desk-level multiple testing is not passed. t_NW 2.90 < 3. DSR 0.31 at n=855 (0.60 at the hunter's 95). The Bonferroni haircut Sharpe
   is 0.43 at n=95 and 0 at n=855.
2. P&L is concentrated in few bets. 61 spells, but the top-3 episodes give 69% of log P&L (one is the 2020-11..2021-04 alt season, +58%).
   With the top-3 removed, SR is 0.59-0.69; ex 10 best days 0.93; ex-2021 0.96; last 18 months 0.47; 2024 -0.14; 2020 0.03.
3. Survivorship flatters the LONG-SPREAD state. It earns +49%/yr on U50, +38%/yr when first-180d coins are excluded, +26%/yr on legacy
   majors. Coins in their first 180 days produced +74% of the +220% arithmetic alt-leg P&L in that state on 16% of the weight. The cleanest
   ex-ante-survival test (signal and basket on legacy majors) gives SR 0.52-0.58, with 2022 negative.
4. At $200 the deployable 11-name book replays at SR 0.47, MaxDD -45%, 2024 -23%. 3,270 legs were skipped below $10, which breaks the
   hedge. It needs at least about $300 (SR 0.89, DD -27%), and even then 2023-24 SR is about 0.15.
My survivorship- and selection-adjusted expectation for forward SR: about 0.5 at full breadth, about 0.3 for the $200 / 11-name book.
P(forward Sharpe > 0.5) is about 0.35 (full breadth about 0.40, $200 11-name book about 0.25).
