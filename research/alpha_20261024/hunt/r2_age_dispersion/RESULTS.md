# r2_age_dispersion — round 2 (HYPE candles only), IS 2020-01-01 .. 2024-12-31

All data came through `lab.py`. Signals use only `close` (plus `qv` for the lab liquidity universe). Universe
`lab.hype_universe("1d", top_k=50)`, cost `lab.HYPE_COST` (0.065%/side), Binance funding charged as cost.
Configs run: family A 42 (38 pre-registered + 4 post-hoc), family B 53 (39 pre-registered + 14 post-hoc), 95 in total.
Scripts: `famA_age.py`, `famA_post.py`, `famB_regime.py`, `famB_post.py`, `famB_post2.py`, `final_numbers.py`
(helpers `r2lib.py`, `ages.py`, `bhelp.py`). Every config is in `runlog.csv`. Grids are in `famA_grid.csv` and `famB_grid.csv`.

## Family A — listing age / supply overhang: NONE (cannot be tested on this data)

Hypothesis: young tokens (low float, unlocks, airdrop selling) underperform mature ones.
Grid: continuous log-age L/S; bucket short [30|60, 180|270|365) vs long mature (>=730d) or vs the rest; raw age
(first Binance perp candle) and "clean" age (old tokens listed late and renames counted as mature); beta-neutral;
vs BTC; Fama-MacBeth.

Result: all 36 trading configs are **negative** (Sharpe -0.54 to -1.59). Young coins *outperformed* mature coins.
In the Fama-MacBeth regression with mom14/vol30/size controls, the age slope is not significant (t -0.9 raw, -0.5 clean).
Best "none": A2_clean_60-270_REST, Sharpe -0.54 (2020-21 -0.81, 2022 -0.19, 2023-24 -0.52), 2x cost -0.58, lag1 -0.55,
neighbour median -0.85. The 16-position version is -0.79.

Why this is not a usable result in either direction: the cache holds only the 150 coins that were alive and
popular in Sep 2026. Only 13 coins listed in 2024 have IS data, while Binance listed far more. The low-float
launches that collapsed are missing. The young bucket is therefore made of survivors (SOL, AVAX, NEAR, OP, PEPE,
WIF...), so the short-young edge is understated and here even reversed. The flipped book (long young / short mature)
shows Sharpe 1.34 with corr to xsmom 0.06. That is a survivorship artefact, not an alpha. The 2025+ forward data uses
the same alive set, so it cannot validate it either. A real test needs delisted symbols (e.g. the full
data.binance.vision archive including dead perps).

## Family B — market-level regime / dispersion timing

Signals: breadth50, Δbreadth10, avg pairwise corr 30/90d, cross-sectional dispersion of 7d returns (disp7), and the
30d mean of daily dispersion (disp30). Each is turned into a trailing 365d percentile, with a MED or TERC (hysteresis)
threshold. Directions were fixed before running: high breadth / low corr / high dispersion = "alt-friendly".

(i) Spread alts-EW vs BTC (±0.5/±0.5, $-neutral):

| config | Sharpe | 20-21 | 2022 | 23-24 |
|---|---|---|---|---|
| **disp7 TERC** | 1.38 | 1.70 | 1.31 | 1.11 |
| disp7 MED | 1.39 | 1.56 | 0.64 | 1.66 |
| corr30 MED | 0.85 | 1.15 | 0.21 | 0.87 |
| breadth50 MED | 0.66 | 1.12 | 0.01 | 0.50 |
| disp30 MED | 0.63 | | | |
| corr90, Δbreadth, TERC variants | -0.06 .. 0.50 | | | |
| control: spread TS-mom 30d / 7d | 0.38 / 0.06 | | | |
| control: BTC-vol timed | 0.48 | | | |

**Candidate B1_disp7_TERC (promising, with caveats).** Signal: disp7 = cross-sectional std of 7d log returns of
the HYPE top-50 at the daily close. p = its percentile in the trailing 365 days. Go long alts EW / short BTC when
p > 2/3, go short alts EW / long BTC when p < 1/3, otherwise keep the previous state. That gives 60 switches in
4.5 years, with an average spell of 27 days.

- Net Sharpe 1.38, CAGR 37.6%, MaxDD -21%, t_NW 2.90.
- Stress: 2x cost 1.28, lag1 1.43 (lag 3: 1.59, lag 5: 1.28, lag 10: 0.76).
- Turnover 39/yr, cost 2.5%/yr, funding -1.3%/yr.
- Both states earn money: long-spread days make +49%/yr (SR 1.64), short-spread days +21%/yr (SR 1.02). So this is
  timing, not a constant alt drift. The mean forward 7d spread is +1.9% in the favourable state and -0.8% otherwise.
- Correlations: BTC 0.03, xsmom14 0.07. Alpha t vs xsmom14 is 2.83, and 2.91 vs [xsmom14, BTC, spread TS-mom 7/30,
  BTC-vol timing].
- Plateau: neighbour median 1.40 (disp3 1.34, disp5 1.47, disp10 1.27, disp14 1.70, pct windows 180/730: 1.40/1.41).
- Universe: top30 1.08, top75 1.53, Binance top50 1.39. Beta-neutral BTC leg 1.63 (post-hoc). Weekly rebalance 1.60.
- DSR 0.68 (53 trials), 0.60 (95 trials).
- Yearly: 2020 -1.6%, 2021 +115%, 2022 +30%, 2023 +63%, 2024 -6%.

Concerns:
- **2024 is flat to negative** in TERC and in every small-book version.
- The edge needs breadth. The 16-position version (15 alts + BTC) gets Sharpe 1.05 (2x cost 0.95, lag1 1.04,
  2024 -17%) and needs $300 at gross 1. The 11-position version ($200 at gross 1) gets 0.91 (2023-24 0.21).
- Survivorship inflates the alt basket. On the 14 legacy majors only it is 0.84 (all sub-periods > 0); with the signal
  also computed on the majors it drops to 0.58.
- With new listings excluded from the dispersion calc, 2023-24 falls to 0.49.
- There are only about 30 regime episodes.

(ii) Overlay on the always-on EW top-50 long basket (Sharpe 0.96, MaxDD -81%):

| overlay | Sharpe | MaxDD | 2022 |
|---|---|---|---|
| trend (EW index > SMA50) | 1.26 | -42% | -0.22 |
| disp7 MED alone | 1.38 | -59% | -0.88 |
| corr30 MED | 1.40 | -65% | -0.67 |
| breadth50 (84% same state as trend) | 0.93 | | |
| **trend AND disp7** | 1.66 | -44% | -0.46 |

The combined overlay has alpha vs the trend overlay of t 1.77 and correlation 0.87 with it. It does not reduce
drawdown beyond plain trend, and 2022 is still negative. Funding costs -12%/yr. Verdict: weak. It is a long-only
alt book that overlaps with RMF's BTC>EMA50 momentum. No regime signal flattens drawdowns better than plain trend.

## Deployability (no VPS, HYPE only)
B1_disp7 needs only HYPE daily closes and volume for the top-50 HYPE coins, so a daily GitHub Actions job can compute
it. It is slow: lag up to 3 days barely hurts. The account size is the binding constraint: at $200 the book must be
about 10 alts + BTC, and that version is clearly weaker.
