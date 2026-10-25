# Verifier pkcc_crit - PKCC30 (Parkinson / close-to-close vol-ratio L/S)

Lens: overfitting, survivorship and risk critic. IS data 2020-01-01..2024-12-31 via lab.py only; ALPHA_FORWARD never set;
no 2025+ data or results read. Costs lab.HYPE_COST (0.065%/side) + funding as cost. Universe lab.hype_universe("1d", top_k=50).
Scripts: s1_multtest.py, s2_stability.py, s3_surv.py, s4_plateau.py, s5_risk.py (all in this folder). Written incrementally.

## Step 0 - independent re-implementation (vlib.py, own code)
- PKCC30 quintile book, HYPE top-50: Sharpe 1.34 (claim 1.34), CAGR +38.9%, vol 27.3%, MaxDD -28.1% (claim -28%), t_NW 3.13, days 1758, start 2020-03-10, avg n_pos 14.3, turnover 106x/yr, skew 0.42, excess kurt 8.3
- sub-periods 2020-21 1.36, 2022 0.75, 2023-24 1.82 (claim 1.36/0.75/1.82); 2x cost 1.09 (1.09); lag1 0.92 (0.92); lag2 0.17 (0.17)
- 16-pos (8/8): Sharpe 1.01 (claim 1.01), MaxDD -34.4%
- Verdict step 0: numbers reproduce -> no implementation discrepancy to chase; critique below is on the statistics, not the code.

## Step 1 - multiple testing
- Sharpe 1.34 over 1758 days (4.82 y): plain t = SR*sqrt(y) = 2.94, Newey-West t = 3.13.
- Expected max Sharpe of null trials (Bailey-LdP): n=49 sd0.5 -> 1.13; n=85 -> 1.24; n=855 sd0.5 -> 1.61, sd0.83 -> 2.66.
- Deflated Sharpe probability (P[true SR > E(max null SR)]):

| book | n=49 sd0.5 | n=49 sd0.83 | n=85 sd0.5 | n=85 sd0.83 | n=855 sd0.5 | n=855 sd0.83 |
|---|---|---|---|---|---|---|
| quintile | 0.67 | 0.11 | 0.59 | 0.06 | 0.28 | 0.00 |
| 16pos | 0.39 | 0.03 | 0.31 | 0.01 | 0.09 | 0.00 |

- Harvey-Liu: t>3 hurdle -> plain t 2.94 FAILS, NW t 3.13 barely passes. At M=855 the Bonferroni hurdle is t>4.02 and BHY(rank 1) t>4.47.
- Haircut Sharpe at M=855: Bonferroni (NW t) -> adj p 1.00, haircut SR 0.00; BHY (NW t) -> adj p 1.00, haircut SR 0.00; Bonferroni (plain t) -> haircut SR 0.00.
- 20d circular block bootstrap (10k): quintile SR 90% CI [0.67, 2.00], median 1.34, P(SR<0.5) = 0.019, P(SR<0) = 0.001.
- 16-pos: 90% CI [0.32, 1.70], P(SR<0.5) = 0.112, P(SR<0) = 0.009.
- quintile lag1: 90% CI [0.25, 1.61], P(SR<0.5) = 0.148.
- NOTE: the bootstrap CI is sampling noise only, conditional on THIS config; it does not include selection (the DSR/haircut lines do).

- Bonferroni t hurdles by effective trial count: M=10 2.81, 49 3.28, 85 3.44, 100 3.48, 200 3.66, 855 4.02. The NW t of 3.13
  clears only M <= ~25 independent trials. Even if the 855 desk configs are heavily correlated (M_eff ~100), it fails.
- Step 1 verdict: sampling noise alone is not the problem (P(SR<0.5) ~2% for the quintile book, ~11% for the 16-pos book),
  but SELECTION is: at the desk's 855 trials the deflated-Sharpe probability is 0.28 (lab default dispersion 0.5) and ~0
  with the empirically observed family-B dispersion (0.83); Bonferroni/BHY haircut Sharpe = 0. The candidate is
  statistically indistinguishable from "best of many noise trials" at desk level. Only at the hunter's own family count
  (n=49, sd 0.5) is DSR > 0.5 (0.67).

## Step 2 - stability
Rolling 365d Sharpe:

| book | min | min date | p10 | median | frac>0 | frac>0.5 |
|---|---|---|---|---|---|---|
| quintile | -0.47 | 2023-08-13 | -0.04 | 1.09 | 89% | 78% |
| 16pos | -0.85 | 2024-01-28 | -0.19 | 0.74 | 79% | 58% |
| quintile_lag1 | -0.70 | 2023-07-28 | -0.22 | 0.72 | 83% | 61% |

- Rolling 365d Sharpe at quarter ends (quintile): 2021Q1 1.38, 2021Q2 1.73, 2021Q3 0.68, 2021Q4 0.86, 2022Q1 1.07, 2022Q2 0.59, 2022Q3 1.18, 2022Q4 0.75, 2023Q1 1.28, 2023Q2 0.40, 2023Q3 0.19, 2023Q4 0.37, 2024Q1 1.19, 2024Q2 2.28, 2024Q3 3.16, 2024Q4 2.88
- Each year alone (quintile): 2020 2.32, 2021 0.86, 2022 0.75, 2023 0.37, 2024 2.95
- Each year alone (16-pos): 2020 2.23, 2021 -0.11, 2022 1.11, 2023 -0.06, 2024 2.70
- Leave-one-year-out (quintile): ex2020 1.13, ex2021 1.66, ex2022 1.47, ex2023 1.51, ex2024 1.02
- ex-2021 1.66; 2023-24 alone 1.82 (16-pos 1.52); 2023 alone 0.37
- Sharpe ex 10 best days: quintile 0.87 (ex 10 best + 10 worst 1.51); 16-pos 0.54 (1.09); lag1 0.44 (1.06). 10 best days = 40% of total summed net return; top 5% days = 184%.
- Top-3 months 2021-04 +25.4%, 2020-07 +16.7%, 2020-11 +15.7% removed -> Sharpe 1.02
- Leave-one-quarter-out: min 1.17 (ex 2021Q2), max 1.59 (ex 2021Q3)
- Leave-one-coin-out (69 coins ever held, universe re-ranked): min 1.16 (ex XLMUSDT), p10 1.25, median 1.33, max 1.40 (ex BTCUSDT). Lowest 5: XLMUSDT 1.16, ETCUSDT 1.17, TRXUSDT 1.18, LINKUSDT 1.18, CRVUSDT 1.19
- Joint removal of top-k price-PnL coins (DOGEUSDT, BCHUSDT, CRVUSDT, XLMUSDT, BNBUSDT ...): top3 0.97, top5 0.86, top10 0.56
- DISP7 regime-episode items: not applicable (PKCC30 has no state variable); month/quarter/coin removals used instead.

- Step 2 verdict: no single coin or quarter carries it (leave-one-coin-out min 1.16), but the time profile is lumpy:
  2020 (2.32, only ~13-15 coins -> 3 per side) and 2024 (2.95) are the strong years; 2021 0.86, 2022 0.75, 2023 0.37.
  Ex-2024 the Sharpe is 1.02, ex-2020 1.13. Rolling 365d Sharpe sat below 0.5 from 2023Q2 to 2023Q4 (min -0.47) and
  then exploded in 2024 (3.2). Fat right tail: 10 best days = 40% of summed net return; ex-10-best Sharpe 0.87
  (16-pos 0.54, lag1 0.44). The 16-pos book (the one $200 can actually trade) is negative in 2021 and 2023.
  Read: the edge is real-looking in aggregate but episodic, and the most recent "normal" year before 2024 (2023) was ~0.

## Step 3 - survivorship
Universe = 150 Binance perps alive Sep 2026, further filtered to the 92 listed on HYPE TODAY (both are future information).

| variant | Sharpe | 2020-21 | 2022 | 2023-24 | lag1 | MaxDD | npos | long-EW xs SR | EW-short xs SR | long-EW %/yr | EW-short %/yr | med age long (d) | med age short (d) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base HYPE top50 | 1.34 | 1.36 | 0.75 | 1.82 | 0.92 | -28% | 14.3 | 0.99 | 1.70 | +34.4% | +52.5% | 653 | 610 |
| a  listed<2022 | 0.88 | 1.36 | 0.44 | 0.46 | 0.61 | -29% | 11.3 | 0.62 | 1.32 | +21.5% | +41.1% | 795 | 731 |
| b1 legacy<2020-03 (15) | 0.66 | 0.77 | 0.25 | 0.76 | 0.35 | -41% | 6.9 | 0.37 | 1.17 | +11.8% | +35.8% | 945 | 935 |
| b2 listed<2021 (27) | 0.82 | 1.32 | 0.54 | 0.24 | 0.38 | -28% | 10.3 | 0.69 | 1.11 | +22.9% | +34.3% | 864 | 912 |
| c  min_hist 180d | 0.81 | 0.73 | 0.56 | 1.26 | 0.45 | -38% | 13.8 | 0.57 | 1.18 | +20.8% | +39.4% | 763 | 728 |
| d  Binance top50 | 1.11 | 0.91 | 0.59 | 1.87 | 0.84 | -40% | 16.3 | 0.95 | 1.35 | +30.7% | +41.3% | 652 | 600 |
| e  Binance top50 non-HYPE | -0.41 | -1.70 | -0.82 | 0.77 | 0.18 | -68% | 2.1 | -0.64 | 0.20 | -35.4% | +13.3% | 550 | 664 |

- Crash coin-days (trailing 5d <= -25%, in universe): held LONG 268 (next-day mean +3.45%), SHORT 144 (+2.65%), not held 613 (+2.90%).
- Coin-days held whose next 10d return <= -40%: long 64, short 48.
- LUNA week 2022-05-07..13 holdings: long ['ADAUSDT', 'ATOMUSDT', 'BCHUSDT', 'BNBUSDT', 'CRVUSDT', 'DOTUSDT', 'GALAUSDT', 'ICPUSDT', 'LINKUSDT', 'LITUSDT', 'LTCUSDT', 'SUSHIUSDT', 'TRXUSDT']; short ['1000SHIBUSDT', 'ARUSDT', 'BCHUSDT', 'DOGEUSDT', 'FILUSDT', 'HBARUSDT', 'ICPUSDT', 'NEARUSDT', 'SANDUSDT', 'SOLUSDT', 'TRXUSDT', 'XMRUSDT', 'XRPUSDT']
- FTX week 2022-11-06..14 holdings: long ['AAVEUSDT', 'ADAUSDT', 'ATOMUSDT', 'AVAXUSDT', 'CRVUSDT', 'DOTUSDT', 'ETHUSDT', 'LINKUSDT', 'LITUSDT', 'LTCUSDT', 'NEARUSDT', 'OPUSDT', 'SOLUSDT', 'SUSHIUSDT']; short ['1000SHIBUSDT', 'ARUSDT', 'BNBUSDT', 'DOGEUSDT', 'FILUSDT', 'GALAUSDT', 'HBARUSDT', 'INJUSDT', 'SANDUSDT', 'TRXUSDT', 'XMRUSDT', 'ZENUSDT']
- SOL during FTX (weight, next-day ret): 2022-11-05 +0.000/-11.6%, 2022-11-06 +0.062/-9.2%, 2022-11-07 +0.062/-19.5%, 2022-11-08 +0.000/-50.9%, 2022-11-09 +0.062/+50.6%, 2022-11-10 +0.062/-6.7%, 2022-11-11 +0.062/-11.4%, 2022-11-12 +0.062/-9.8%, 2022-11-13 +0.062/+6.7%, 2022-11-14 +0.062/+2.2%, 2022-11-15 +0.062/+nan%

### Step 3b - survivorship follow-ups
Price-PnL share of the base book by coin age (share of total price PnL; share of held coin-days):

| age bucket | long-leg PnL share | short-leg PnL share | share of coin-days |
|---|---|---|---|
| <180d | +71% | -27% | 13% |
| 180-365d | +51% | -9% | 18% |
| >365d | +43% | -28% | 69% |

- Crash coin-days (trailing 5d <= -25%): long-leg PnL share +34%, short-leg -17% of total price PnL.
- Universe minus crash coin-days (5d<=-25%): Sharpe 1.02; (5d<=-20%): 1.09; min_hist180 AND no crash coin-days: 0.63
- Non-HYPE survivors ranked among themselves, top-30: Sharpe -0.52 (sub 2020-21 -1.70, 2022 -0.82, 2023-24 0.20), npos 3.7; top-50: -0.52. Like-for-like HYPE top-30: 1.13.
- SOL closes 2022-11-04..12: 2022-11-04 33.72, 2022-11-05 36.89, 2022-11-06 32.60, 2022-11-07 29.59, 2022-11-08 23.82, 2022-11-09 11.70, 2022-11-10 17.61, 2022-11-11 16.43, 2022-11-12 14.56

- Note on the non-HYPE test: only 20 non-HYPE survivors have IS data (avg 3.7 positions), so the -0.52 is thin evidence; it
  is still the wrong sign. Binance top-50 (HYPE filter removed) = 1.11 vs 1.34: the "listed on HYPE today" filter
  (future information) adds ~0.23 Sharpe.
- Step 3 verdict (survivorship, stated plainly):
  * The edge lives disproportionately in YOUNG coins: coin-days in a coin's first 180 days are 13% of the book but earn
    ~44% of total price PnL (long +71%, short -27%). Dropping them (min_hist 180d) cuts Sharpe 1.34 -> 0.81 and lag1
    0.92 -> 0.45. Restricting to coins listed before 2022 -> 0.88 with 2023-24 collapsing 1.82 -> 0.46; legacy majors
    only -> 0.66 (lag1 0.35). The 2023-24 strength is a 2023-24-listing-cohort effect, i.e. exactly where survivor
    selection (still alive AND listed on HYPE in 2026) is strongest.
  * The LONG leg is the flattered one. Long-minus-EW excess Sharpe 0.99 -> 0.57 (min_hist 180) / 0.62 (pre-2022)
    / 0.37 (legacy), i.e. -40% to -60%; ~+13 pts/yr of the long excess disappears. The short excess also falls
    (1.70 -> 1.18 / 1.32 / 1.17, -25% to -30%), so the hunter's "short leg carries it, so survivorship is conservative"
    is only half right: the short excess is more robust, but it too shrinks outside young survivors.
  * Mechanism: a low PK/CC ratio means closes at the day's extreme, which is what a CRASH looks like. The long leg holds
    crash coin-days (trailing 5d <= -25%) almost 2x as often as the short leg (268 vs 144); among survivors these bounce
    (+3.45% next day) and contribute +34% of total price PnL. The coins that crashed and DIED (LUNA 2022-05, FTT/SRM
    2022-11, ...) are exactly the ones missing from the data and would have sat in the long leg. Excluding crash
    coin-days -> Sharpe 1.02; excluding them AND young coins -> 0.63. A single LUNA-style wipe in the long leg costs
    ~0.5/7 x 90% = ~6.5% of equity.
  * Survivorship-adjusted central estimate of IS Sharpe: ~0.6-0.9, not 1.34.

## Step 4 - plateau (20 pre-registered cells, see s4_plateau.py header)

| cell | Sharpe | lag1 | 2x cost | 2020-21 | 2022 | 2023-24 |
|---|---|---|---|---|---|---|
| W20_q0.15 | 1.12 | 0.72 | 0.88 | 0.95 | 1.45 | 1.71 |
| W20_q0.20 | 1.22 | 0.84 | 0.96 | 1.04 | 1.05 | 2.06 |
| W20_q0.30 | 0.94 | 0.82 | 0.65 | 0.88 | 0.48 | 1.58 |
| W25_q0.15 | 0.99 | 1.44 | 0.77 | 0.88 | 1.32 | 1.39 |
| W25_q0.20 | 1.14 | 1.30 | 0.91 | 0.94 | 1.25 | 1.87 |
| W25_q0.30 | 0.98 | 1.09 | 0.72 | 0.59 | 1.22 | 1.82 |
| W30_q0.15 | 1.39 | 0.85 | 1.16 | 1.70 | 0.67 | 1.50 |
| W30_q0.20 | 1.34 | 0.92 | 1.09 | 1.36 | 0.75 | 1.82 |
| W30_q0.30 | 1.12 | 0.64 | 0.86 | 0.69 | 0.97 | 2.12 |
| W40_q0.15 | 1.09 | 0.80 | 0.88 | 1.34 | 0.44 | 1.19 |
| W40_q0.20 | 1.33 | 0.89 | 1.11 | 1.50 | 0.70 | 1.56 |
| W40_q0.30 | 1.26 | 0.84 | 1.02 | 1.07 | 0.70 | 1.99 |
| W45_q0.15 | 1.43 | 0.95 | 1.24 | 1.90 | 0.60 | 1.31 |
| W45_q0.20 | 1.52 | 1.00 | 1.31 | 1.96 | 0.62 | 1.48 |
| W45_q0.30 | 1.19 | 0.90 | 0.95 | 1.21 | 0.46 | 1.67 |
| W30_q0.20_top40 | 1.08 | 0.84 | 0.83 | 1.36 | 0.71 | 0.96 |
| W30_q0.20_top60 | 1.35 | 0.97 | 1.10 | 1.36 | 0.75 | 1.87 |
| GK30_q0.20 | 1.35 | 0.91 | 1.09 | 1.35 | 0.76 | 1.85 |
| EWMA_hl15_q0.20 | 1.53 | 1.10 | 1.31 | 1.73 | 0.49 | 1.98 |
| W30_rankprop | 1.16 | 0.83 | 0.90 | 0.97 | 0.87 | 1.83 |

- Sharpe: median 1.20, IQR [1.11, 1.35], min 0.94, max 1.53, share>=1.0 85%, share>=0.6 100%
- lag1: median 0.90, IQR [0.84, 0.98], min 0.64, max 1.44, share>=1.0 25%, share>=0.6 100%
- 2x cost: median 0.95, IQR [0.88, 1.10], min 0.65, max 1.31, share>=1.0 45%, share>=0.6 100%
- min sub-period Sharpe per cell: median 0.70; cells with every sub-period > 0: 20/20
- Secondary, survivorship-crossed (min_hist 180d, q0.2): W20_q0.20_hist180 0.44/lag1 0.13, W25_q0.20_hist180 0.42/lag1 0.90, W30_q0.20_hist180 0.81/lag1 0.45, W40_q0.20_hist180 1.00/lag1 0.88, W45_q0.20_hist180 1.00/lag1 0.96 -> median 0.81, lag1 median 0.88

- Step 4 verdict: the LOCAL surface (W 20-45, q 0.15-0.30, top_k 40-60, GK/EWMA/rank-weight variants) is a genuine
  plateau, not a spike: median Sharpe 1.20 (IQR 1.11-1.35), lag1 median 0.90, 2x cost median 0.95, every cell has all
  three sub-periods > 0. This is the strongest point in the candidate's favour. Caveats: (i) the hunter's wider
  pre-registered surface is much less flat (W10 0.66, W60 0.64, W90 0.23; R5/R7 holding 0.40/0.33), so the plateau is
  ~20-45d wide and daily-rebalance-only; (ii) the same W-sweep on the survivorship-robust universe (no coin in its first
  180 days) has median 0.81 (0.42-1.00), so the plateau level is inflated by young survivors, consistent with step 3.

## Step 5 - risk
- alignment sanity corr(long leg, EW universe, BTC): L~EW 0.93, L~BTC 0.77
- Worst day / 7d / month, quintile: -9.4% (2021-02-13), -19.2% (7d to 2021-02-14), -13.1% (2021-08). 16-pos: -8.2% (2021-02-13), -14.8% (2021-02-14), -13.0% (2021-01).
- Worst drawdown (quintile): -28.1%, peak 2021-01-07 -> trough 2021-02-16 (40 d), recovered 2021-04-16. Over the episode: long leg +32.7%, short leg -46.8%, BTC +33.6%, EW universe +206.7%. Worst coin contributions (price PnL, fraction of equity): XRPUSDT -8.1%, ZENUSDT -7.8%, ETCUSDT -7.1%, TRXUSDT -6.7%, SUSHIUSDT -6.6%, AAVEUSDT -5.8%
- 2nd-worst distinct episode: -26.7%, 2021-06-23 -> 2021-10-04.

| crash window | quintile | 16-pos | long leg | short leg | BTC | EW univ |
|---|---|---|---|---|---|---|
| COVID (book starts 03-10) | -0.2% | -1.6% | -10.0% | +3.7% | -21.8% | -25.4% |
| May-2021 crash (05-19) | -9.5% | -13.0% | -22.7% | +8.4% | -34.3% | -39.5% |
| 2021-12-04 flush | -1.2% | -2.2% | -9.5% | +8.8% | -10.8% | -17.1% |
| LUNA | +7.1% | +5.1% | -8.2% | +13.9% | -14.3% | -25.3% |
| 3AC/Celsius | -2.8% | -3.6% | -12.1% | +9.7% | -31.6% | -23.1% |
| FTX | +0.9% | +1.0% | -13.3% | +13.4% | -20.7% | -27.0% |
| SVB/USDC | +1.3% | +1.4% | +5.0% | -3.8% | +13.8% | +11.7% |
| Apr-2024 flush | +3.7% | +5.6% | -8.3% | +11.9% | -9.5% | -21.2% |
| yen carry 08-05 | +3.9% | +2.1% | -9.4% | +14.0% | -14.7% | -22.3% |
| Dec-2024 alt flush | +2.3% | +2.2% | -4.6% | +6.4% | +1.2% | -9.5% |

- 20 worst BTC days (mean BTC -12.5%): strategy mean +0.05%, hit rate 40%, worst -6.1%. 20 worst EW-alt days (mean -17.5%): strategy mean +0.34%, hit 35%, worst -3.7%.
- BTC beta: all +0.039, BTC-up days +0.058, BTC-down days +0.004, BTC < -5% days +0.020; beta to EW universe +0.032.

| causal state | days | Sharpe | ann. mean | BTC beta | 16-pos Sharpe |
|---|---|---|---|---|---|
| BTC>200dSMA = True | 1061 | 1.50 | +46.4% | +0.046 | 1.11 |
| BTC>200dSMA = False | 697 | 1.05 | +21.6% | +0.032 | 0.86 |
| BTC hi-vol = True | 613 | 0.73 | +24.6% | +0.045 | 0.61 |
| BTC hi-vol = False | 1145 | 1.86 | +43.0% | +0.035 | 1.35 |
| BTC beats EW alts 30d (dominance up) = True | 871 | 1.88 | +43.3% | +0.045 | 2.01 |
| BTC beats EW alts 30d (dominance up) = False | 887 | 0.97 | +30.0% | +0.035 | 0.32 |

- $200 account sim (account_sim.py, $10 min order, HYPE cost + funding, 2020-03..2024-12):
  * quintile: end equity $893, MaxDD -30.0%, min equity $187, Sharpe 1.22, orders 6717, positions skipped below $10: 0, fees $138
  * 16pos: end equity $583, MaxDD -35.3%, min equity $187, Sharpe 0.97, orders 6528, positions skipped below $10: 0, fees $95

- Step 5 verdict (risk): market-neutral in practice (BTC beta +0.04 overall, +0.00 on BTC-down days, +0.02 on BTC<-5%
  days) and it HEDGES most alt crashes (LUNA +7.1%, FTX +0.9%, Apr-24 +3.7%, Aug-24 +3.9%) because the short leg
  (wicky alts) falls more than the long leg. Exceptions: May-2021 crash -9.5% (16-pos -13%) and 3AC -2.8%.
  The real risk is the opposite tail, an ALT-SEASON SQUEEZE of the short leg: worst DD -28.1% from 2021-01-07 to
  2021-02-16 while the EW universe rose +207% (short leg -47%); 2nd-worst -26.7% Jun-Oct 2021; worst day -9.4% and
  worst week -19.2% both on 2021-02-13/14. By causal state: Sharpe 1.88 when BTC beat the EW alts over the prior 30d
  ("dominance up") vs 0.97 when alts led (16-pos: 2.01 vs 0.32); 1.86 in low BTC vol vs 0.73 in high BTC vol.
  $200 account sim: quintile $200 -> $893 (MaxDD -30%), 16-pos -> $583 (MaxDD -35%); no orders skipped at $10 min.

### Step 6 diagnostics (rationale; not selection)
- PnL split by trailing-5d sign (share of total price PnL): 5d up: long +112% short -49%; 5d down: long +54% short -18%
- XS spearman pkcc30 vs mean30|CLV| -0.29, vs max30|r| -0.44, vs |r_t| -0.14
- long high CC-vol30 only: Sharpe 1.05
- long low PK-vol30 only: Sharpe -0.97
- long high mean30|CLV|: Sharpe 0.44

- corr(PKCC30, long-high-CCvol30 book) 0.42; CCvol30 book with min_hist180: Sharpe 0.58; PKCC30 on [CCvol30, xsmom14, BTC]: alpha +18.9%/yr t 1.89, loading CCvol +0.28 (t 5.4), xsmom +0.18

## Step 6 - economic rationale critique
- What the signal really selects. PK/CC < 1 when daily closes sit at the day's extreme (|close-to-close move| large vs
  the range); the 30d mean of cc^2 is dominated by the few biggest |return| days. So the LONG leg = coins that recently
  had big one-directional days (pumps AND crashes, it is unsigned); the SHORT leg = coins with wide ranges but little net
  movement (wicky/choppy, typically mid-cap alts being stop-hunted). Diagnostics: XS Spearman of the ratio with max30|r|
  is -0.44; a plain "long high close-to-close vol30" book alone earns Sharpe 1.05 (0.58 without young coins), correlates
  0.42 with PKCC30, and after adding it (plus xsmom14 and BTC) PKCC alpha is +19%/yr with t 1.89 (< 2). Long-leg PnL
  comes from 5d-up coins (+112% of total price PnL) and 5d-down coins (+54%); the short leg loses in both.
- Plausible persistence story (hunter's): clean moves = informed/persistent flow that keeps going for 1-2 days; wicky
  ranges = absorbed noise flow / stop runs that lag. This is a microstructure continuation effect, consistent with the
  2-day decay (lag2 0.17). Microstructure effects of this kind are usually real but small and crowd out fast; nothing
  here is a risk premium that someone is paid to bear, so there is no structural reason for it to persist.
- Competing (less flattering) story: part of the long leg is "lottery/high-realised-vol survivors" and "crash-bounce
  survivors" - both are textbook survivorship artefacts in a universe of coins known to be alive in 2026 (step 3: crash
  coin-days +34% of PnL, coins <180d old +44% of PnL). Live, the coins that crash and do not bounce (LUNA/FTT type) will
  be in the long leg.
- When it would stop working: (1) alt-season squeezes - IS Sharpe 0.97 (16-pos 0.32) when alts beat BTC over 30d and
  the worst DDs are Jan-Feb and Jun-Oct 2021; (2) high BTC vol regimes (0.73); (3) if the wicky short-leg coins become
  the momentum leaders (memecoin manias). A BTC-dominance / ETF-flow regime is NOT the threat IS: dominance-up periods
  were the best (1.88) and 2024 (ETF year) was the best year (2.95). The threat is the opposite regime, a broad alt
  rotation, plus crowding of a fast 1-2 day signal and execution slippage at 00:00 UTC on small alts.
- Step 6 verdict: rationale is plausible but post hoc and partly confounded with survivorship-flattered components.

## Summary (pkcc_crit)
- Code: reproduces exactly (1.34 / -28% / 1.36-0.75-1.82 / 2x 1.09 / lag1 0.92 / lag2 0.17 / 16-pos 1.01).
- For: genuine local plateau (20 pre-registered cells, median 1.20, lag1 0.90, 2x cost 0.95, all sub-periods > 0);
  near-zero BTC beta, hedges most alt crashes; no single coin/quarter dependence; bootstrap P(SR<0.5) ~2%.
- Against: fails desk-level multiple testing (DSR 0.28 at n=855, Bonferroni haircut 0, plain t 2.94 < 3);
  survivorship: no-young-coins 0.81 (lag1 0.45), pre-2022 listings 0.88 (2023-24 0.46), legacy majors 0.66;
  long leg is the flattered leg (crash-bounce + young-survivor PnL); 2023 ~0 and the deployable 16-pos book negative in
  2021 and 2023; ex-10-best-days 0.87; alpha t 1.89 after a high-CC-vol book is added.
- Verdict: confirmed_with_concerns. Survivorship- and selection-adjusted IS Sharpe ~0.6-0.9; expected live ~0.4-0.6.
  P(forward Sharpe > 0.5) ~ 0.40 (quintile book); ~0.33 for the 16-pos book.
