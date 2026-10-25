# r2_seasonality — round 2 results (IS 2020-01-01 .. 2024-12-31, lab.py only)

**Bottom line: no edge to forward in either family.** Every signal uses only close prices and the
calendar, so all of it could run on HYPE candles from a daily or 4h GitHub Actions job. No VPS is needed and
no funding, OI or ratio inputs are used. Funding is charged only as a cost.

Cost is lab.HYPE_COST (0.065%/side) plus funding. Universe is `lab.hype_universe("1d", top_k=50)`. In total
73 configs were run: family A 46 (A1 19, A2 19, A3 8) and family B 27.

## Family A — cross-sectional seasonality (Heston-Sadka), 46 configs

| sub-family | what | best net Sharpe | gross Sharpe | turnover/yr | verdict |
|---|---|---|---|---|---|
| A1 weekday (1d) | mean return on the same weekday over the last N weeks, N 8/13/26/52 | -0.47 (N13 raw 8L/8S) | +0.83 | ~540 | none: no weekday seasonality |
| A2 4h slot (4h) | mean return in the same 4h slot over the last N days, N 7/14/30/60 | -5.0 (N60 raw 8L/8S) | **+3.44** | ~3200 | none: real effect, but cost wipes it out |
| A3 session blocks | the same idea on 8h/12h blocks, N 30/60 | -0.20 (12h N30 raw) | +2.39 | ~1100 | none: and 0.54 correlated with momentum |

- **A1.** The raw same-weekday score has a rank-IC of about 0 (t from -0.4 to +0.9). The "excess" score has a
  positive IC (t ≈ 3), but that comes from its "other days" part, not from the same-weekday part. The
  other-days placebo has IC -0.025 (t ≈ -4), which is plain reversal of trailing returns. Net Sharpe is
  between -0.5 and -1.7 in all 16 core configs, and costs are 31–38%/yr.
- **A2 is a real anomaly that cannot be traded at taker cost.**
  - The raw same-slot IC is +0.018 (t 8.4). Gross is about +85%/yr, with gross alpha t of 6.6 against
    momentum and BTC. Gross return is positive every year.
  - The effect is tied to the slot. Lagging the book 24h keeps gross Sharpe at +3.7, while lagging it 4h or
    8h turns gross Sharpe to about -1.9. So coin-specific intraday shapes (Asia vs US hours) persist.
  - The excess version is not momentum: correlation 0.03, gross Sharpe 3.2.
  - But the book rotates completely every 4h. Break-even cost is **2.6 bp/side**, against 6.5 bp on HYPE.
    Net Sharpe is between -5 and -10 in all 16 configs.
  - Grouping into 8h/12h blocks (A3) cuts turnover by 2–3x, but it also cuts the signal and adds momentum
    exposure.
  - It could only pay with maker fills at no more than about 2 bp all-in. That does not fit a cron job with
    no VPS.
- Robustness checks (top_k 30/75, q=0.2) on the best A1 and A2 configs change nothing.

## Family B — calendar effects at low turnover, 27 configs

The rule was set before running: estimate on 2020–22, validate on 2023–24. A config passes if EST t ≥ 1.5
and VAL t ≥ 1.0 with the same sign. **0 of 24 passed.**

- **Turn-of-month (TOM) was strong in 2020–22, then reversed in 2023–24.**
  - ETH TOM33 (last 3 days plus first 3 days of the month): +128 bp/day in the window vs +11 bp outside
    (t 3.2) in EST. In VAL it was -18 bp vs +27 bp (t -1.9).
  - For the EW basket, TOM episodes were positive in 79% of months in EST, with a median of +5.5%. In VAL
    only 44% were positive, with a median of -1.3%.
  - Over the full IS, ETH TOM33 has Sharpe 1.34, 1.29 at 2x cost and 1.30 at lag 1. But by sub-period it is
    2.82 / 0.26 / **-0.84**. By year it made +450% in 2021 and -41% in 2024.
  - This looks like an effect of the 2020–21 retail boom, or just noise, not a persistent edge.
- **Month-end contrarian rebalancing (MEREV):** negative in both halves for all assets.
- **Monday (Sunday close to Monday close):** BTC and ETH show the same sign in both halves (t ≈ 0.9),
  but it is weak and loses to buy & hold. Round 1 already covered this.
  - Alts underperform BTC on Mondays in both halves (t -1.7 / -0.9). Trading that flip would net about 0
    after costs. It is post-hoc, so I did not pursue it.
- **Long-alt / short-BTC spread versions:** EST t was 1.8–1.9 for TOM, and VAL was about 0.
- **Robustness (TOM33 EW):** the 16-name basket, top-30 and top-75 all give the same picture, with 2023–24
  Sharpe between -0.5 and -0.7.

## Exposure notes

- **Survivorship.** Family A is dollar-neutral, so the short leg is understated: coins that later died are
  missing. Family B's long basket is inflated, but the in-window vs out-of-window test spreads that bias over
  all days.
- **Overlap with RMF/RNT.** Correlation with 14d XS momentum is: A1 0.07, A2 raw 0.24 / excess 0.03,
  A3 0.54, B about 0.01.

## Files

- `common.py`: helpers for evaluation, the alpha regression and the 8L/8S weights.
- `famA_xs_seasonality.py`: A1 and A2 grid, robustness, IC and placebo. Output in `famA_grid.csv` and
  `famA_ic.csv`.
- `famA_diag.py`: break-even cost, lag profile and per-slot split.
- `famA3_session.py`: A3 session blocks.
- `famB_calendar.py`: B grid with the EST/VAL rule, output in `famB_grid.csv`.
- `famB_robust_and_summary.py`: robustness and the summary table in `all_configs.csv`.
- `famB_tom_concentration.py`: TOM returns per episode.
- `*.log`: the run outputs.
