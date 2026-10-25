# PKCC30 verification — lens: independent reimplementation + lookahead audit + $200 design

Verifier: pkcc_impl (2026-10-24). Data: lab.load (truncated < 2025-01-01). ALPHA_FORWARD never set.
Desk multiple-testing count: ~855 configs (both rounds). This verifier's own configs are counted below.

## Step 1 — independent reimplementation (rules text only, hunter code not read yet)

Script: s1_reimpl.py (+ common_pk.py). Own dollar-equity loop: trade W[t] at close t, earn close t->t+1,
pay Binance funding settled in bar t+1, fee 0.065%/side on |target - drifted holding|. Cross-check: same W
through lab.backtest gives identical result (corr 1.00000, max abs daily diff 7.6e-6).
Signal: pk=(ln H/L)^2/(4ln2), cc=(ln C_t/C_t-1)^2, joint obs only, rolling mean n=30 min 24, ratio=sqrt(mpk/mcc).
Selection: pandas pct-rank among U & ratio-valid; long rank<=0.2 (LOW ratio), short rank>=0.8 (HIGH ratio), EW 0.5/side.

| config | own Sharpe | hunter claim | gap |
|---|---|---|---|
| base n30 | 1.20 (CAGR +35.7%, MDD -33.5%) | 1.34 (MDD -28%) | -0.14 |
| sub 2020-21 / 2022 / 2023-24 | 1.24 / 0.91 / 1.49 | 1.36 / 0.75 / 1.82 | -0.12 / +0.16 / -0.33 |
| 2x cost | 0.96 | 1.09 | -0.13 |
| lag1 | 0.84 | 0.92 | -0.08 |
| lag2 | 0.58 | 0.17 | +0.41 |
| window 10/20/30/45/60/90 | 0.66/1.22/1.20/1.10/0.59/0.05 | 0.66/1.22/1.34/1.52/0.64/0.23 | 0/0/-.14/-.42/-.05/-.18 |
| k=8/side (16 pos) | 0.98, lag1 0.54 | 1.01, lag1 0.55 | ~0 |

Yearly (own, base): 2020 +81.7% (SR 2.66) | 2021 +15.9% (0.55) | 2022 +19.7% (0.91) | 2023 -2.6% (-0.08) | 2024 +77.1% (2.68).
Other: turnover 105x/yr, cost drag 6.8%/yr, funding +1.6%/yr (received, net-short tilt); no-funding SR 1.14.
Positions/day mean 14.2 (long 6.6 / short 7.6, min 3, max 21); eligible names/day 16 (2020) -> 49 (2024).
First trade 2020-03-09. Corr with BTC 0.10, with 14d XS momentum 0.33.
Raw legs (each 0.5 gross, NOT market-neutral): long-only SR 1.33 (2022 -0.86), short-only SR -0.69 (MDD -90%).
-> "short leg carries the edge" must be relative-to-universe; checked in step 1b.
Own configs run in step 1: 14.

### Step 1b–1d — gap explanation (after own numbers were fixed, hunter code read: signals.py, common.py, final_numbers.py)

Hunter's exact W rebuilt and run through MY loop: SR 1.34, sub 1.36/0.75/1.82, 2x 1.09, lag1 0.92,
yearly +65/+31/+15/+5/+86% -> hunter's numbers reproduce exactly (no arithmetic error in lab/hunter).
The 0.14 gap has two sources, neither is a bug in my code, both are fragility findings:

1. **Leg-count convention (main source).** pandas pct-rank: "top q" (r>=0.8) takes floor(0.2N)+1 names,
   "bottom q" (r<=0.2) takes floor(0.2N). The hunter ranks -ratio, so he holds floor+1 LOW-ratio longs and floor
   HIGH-ratio shorts; my literal reading did the reverse. All four readings of "bottom/top 20%" (s1d_legcount.py):

| convention (nL/nS avg) | SR | MDD | 2020-21/2022/2023-24 | 2x | lag1 | 2021 ret |
|---|---|---|---|---|---|---|
| A own: L=floor, S=floor+1 (6.6/7.6) | 1.20 | -33.5% | 1.24/0.91/1.49 | 0.96 | 0.84 | +16% |
| B hunter: L=floor+1, S=floor (7.6/6.6) | 1.31 | -28.1% | 1.36/0.59/1.82 | 1.06 | 0.89 | +31% |
| C both floor (6.6/6.6) | 1.41 | -27.1% | 1.77/0.72/1.40 | 1.17 | 1.04 | +55% |
| D both floor+1 (7.6/7.6) | 1.08 | -35.8% | 0.77/0.79/1.93 | 0.83 | 0.66 | -2% |

   One name per side moves full-sample Sharpe across 1.08-1.41 and the 2021 return across -2%..+55%.
   Adding names (D) HURTS: the edge sits in the most extreme ~6 names per side, consistent with 16-pos = 1.01.
2. **Stale-data artefact (2022 only, +0.03 full / +0.16 in 2022).** ICPUSDT has 82 days with H==L (frozen
   price) in Jun–Aug 2022 while still inside the universe. Hunter clips H/L at 1+1e-5 so ICP's ratio = +inf ->
   permanently in the short leg at zero return (an idle short slot = smaller effective short exposure during the
   Jul-2022 bounce). My version gives 0/0 = NaN and excludes it. Also a 3-day data gap (2022-02-26..28, 13 coins)
   is handled slightly differently (joint vs separate obs). B (no artefact) 2022 SR 0.59 vs hunter 0.75.

Leg excess vs EW universe (each leg scaled to gross 1): hunter convention long 0.95-0.99 / short 1.70;
own convention long 1.09 / short 1.31. "Short leg carries the edge" depends on the convention; both legs add.
Configs run in step 1b-d: 5 (step-1 total 19).

## Step 2 — lookahead audit (s2_audit.py)

**(a) Truncation test — PASS.** Data panels (high, low, close, qv, fund) cut at 2021-05-19, 2022-11-08, 2024-03-14
and everything rebuilt (universe, signal, weights, returns) for both leg conventions. Universe cells, signal cells,
weights (incl. the last row before each cut, 13/15/19 positions) and all daily returns before the cut are
bit-identical to the full-sample run (max diff 0.0). No future information reaches the signal or the book.

**(b) Static audit.**
- Windows: plain trailing rolling(30, min 24) on pk and cc; no centering, no full-sample z-score/rank; the only
  ranking is cross-sectional per day. Universe = trailing 30d median qv rank + listing age >= 60d (trailing).
- Listing-day candles never enter a traded window (age >= 60 > 30). 271 of 25,023 book cells use < 30 joint obs
  (data gaps, e.g. 2022-02-26..28 missing for 13 coins). Harmless.
- First valid/traded date 2020-03-09. **2020 is thin: hunter convention holds 4.1 long / 3.1 short on average
  (min 2 / 1 names)**; 2021 6.6/5.6; 2022 7.2/6.2; 2023 9.0/8.0; 2024 10.5/9.5. The best year (2020, +65..82%)
  is a 3-4-name book.
- Stale data: ICPUSDT frozen (H==L) for 82 days in mid-2022 while inside the universe; 29 book cells hold a coin on
  an H==L day. In the hunter's code this coin becomes ratio=+inf and is forced into the short leg (see step 1).
  Not lookahead, but a data artefact that the hunter's number benefits from (+0.16 Sharpe in 2022).
- lab.backtest's "flat if no next-bar return" uses one bar of future availability; only bites at data gaps here.
- Selection bias (not price lookahead): universe = coins alive in 2026 AND listed on HYPE in 2026, applied back to
  2020. Long leg (clean movers) is the side that survivorship flatters.

**(c) Execution delay (1h closes, hold 24h from 00:00+dh):**

| delay | own conv A | hunter conv B |
|---|---|---|
| 0h | 1.20 | 1.31 |
| 1h | 1.20 | 1.32 |
| 2h | 1.14 | 1.29 |
| 4h | 1.06 | 1.19 |

A cron that runs 5-60 min late costs nothing; 4h late costs ~0.13. extra_lag=1 day (step 1) costs 0.36-0.42.
Configs in step 2: 6 (delay) — running total 25.

## Step 3 — $200 design: PRE-REGISTRATION (written before any step-3 run)

Fixed for all configs: universe U_t = lab.hype_universe("1d", top_k=50) & ratio valid, N_t = #names;
signal = PKCC ratio n=30 (min 24 joint obs), unchanged; long LOWEST ratio, short HIGHEST ratio; equal weight,
each name G/(2k); $-neutral. Names per side k_t = min(floor(E_t*G/2/12.5), kmax, floor(N_t/4)); if k_t < 3 -> flat.
(E_t = account equity at close t; 12.5 USD target per name gives a 25% buffer above the 10 USD minimum.)
Simulator: account_sim mechanics (equity0=200, min_order=10, min_trade=10, cost=lab.HYPE_COST, funding charged);
names chosen inside a dynamic loop (k depends on equity), the realised weight panel is then replayed through
account_sim.simulate to confirm identical equity. Each config: full 2020-03-01..2024-12-31, fresh $200 on
2023-01-01, and lag-1 (selection from close t-1) as a stress (stress runs are not separate configs).

| id | G | kmax | rebalance | band |
|---|---|---|---|---|
| D1 | 1.00 | 8 | daily | none |
| D2 | 1.25 | 8 | daily | none |
| D3 | 1.50 | 8 | daily | none |
| D4 | 1.00 | 6 | daily | none |
| D5 | 1.25 | 6 | daily | none |
| D6 | 1.50 | 6 | daily | none |
| D7 | 1.25 | floor(0.2N) (quintile) | daily | none |
| D8 | 1.25 | 8 | daily | rank hysteresis: keep held name while rank <= k+3, fill to k by best rank |
| D9 | 1.25 | 6 | daily | rank hysteresis +3 |
| D10 | 1.25 | 8 | every 2 days | none |
| D11 | 1.25 | 8 | weekly (7d) | none (control, expected to fail) |
| D12 | 1.00 | floor(0.2N) (quintile) | daily | none |

Pre-registered selection rule (robustness, not peak): score = min(Sharpe full, Sharpe fresh-2023, Sharpe lag-1 full);
eligible only if final equity > 200 in both full and fresh-2023 runs and no liquidation-like path (equity never < 80).
Highest score wins; ties (within 0.05) -> lower full-period MaxDD, then fewer orders/day.

## Step 3 — results (s3_design.py; partial write while grid runs, completed below)

All runs: $200 start, $10 min order/trade, HYPE_COST, funding charged; realised weights replayed through
account_sim.simulate -> identical final equity in every run (replay True). "2025:-0%" in logs = the forced
close-out fee booked on the last IS bar (index 2025-01-01 = end of 2024-12-31 bar); no 2025 data used.

| id | full SR (frictionless) | $200 -> | MDD | min eq | 2020/21/22/23/24 | fresh-2023 SR, $200 -> , MDD | lag1 SR | score |
|---|---|---|---|---|---|---|---|---|
| D1 G1.0 k8 | 0.85 (0.96) | 523 | -37% | 186 | +58/-14/+11/-2/+77 | 1.42, 359, -24% | 0.41 | 0.41 |
| D2 G1.25 k8 | 0.85 (0.95) | 618 | -46% | 182 | +70/-24/+16/-2/+111 | 1.44, 405, -26% | 0.45 | 0.45 |
| D3 G1.5 k8 | 0.89 (0.95) | 765 | -51% | 182 | +90/-26/+18/-4/+141 | 1.45, 459, -31% | 0.45 | 0.45 |
| D4 G1.0 k6 | 1.01 (1.10) | 692 | -34% | 186 | +58/+22/+4/-7/+87 | 1.27, 353, -26% | 0.62 | 0.62 |
| D5 G1.25 k6 | 1.01 (1.10) | 881 | -38% | 182 | +70/+21/+8/-9/+121 | 1.22, 383, -33% | 0.62 | 0.62 |
| D6 G1.5 k6 | 1.03 (1.10) | 1123 | -44% | 182 | +90/+18/+9/-12/+158 | 1.22, 427, -38% | 0.68 | 0.68 |
| D7 G1.25 kq | 1.35 (1.38) | 1615 | -34% | 198 | +127/+60/+18/-5/+98 | 1.34, 377, -31% | 0.95 | 0.95 |

Note D5 vs D7 differ only in the per-side count rule (6 vs floor(0.2N), both capped at N//4): 2020 uses 4 vs 3 names,
2021 6 vs 5-7 names — and 2021 swings +21% vs +60%. Same one-name fragility as step 1.
| D8 G1.25 k8 band3 | 0.97 (1.06) | 715 | -40% | 177 | +52/+15/-0/-2/+109 | 1.44, 415, -32% | 0.60 | 0.60 |
| D9 G1.25 k6 band3 | 0.98 (1.05) | 792 | -40% | 177 | +52/+51/-7/-6/+98 | 1.24, 384, -36% | 0.60 | 0.60 |
| D10 G1.25 k8 reb2 | 0.80 (0.85) | 568 | -42% | 182 | +73/-8/+6/+6/+58 | 1.01, 322, -22% | 0.42 | 0.42 |
| D11 G1.25 k8 reb7 | 0.46 (0.48) | 322 | -54% | 147 | +44/-29/+5/+11/+35 | 0.85, 294, -23% | -0.31 ($200->82) | -0.31 |
| **D12 G1.0 kq** | **1.32 (1.38)** | **1094** | **-29%** | 198 | +94/+49/+11/-4/+78 | **1.38, 350, -24%** | **1.00** | **1.00** |

Orders/day 1.6-4.4; no order was ever skipped for being < $10 (k rule keeps targets >= $12.5).
Hysteresis (D8/D9) halves orders and fees (2.0 vs 3.7 orders/day) and lifts the k8 book (0.85 -> 0.97) but not k6.
Weekly (D11) and 2-day (D10) rebalance confirm the signal must be traded daily.
**Every daily config loses money in 2023 (-2% .. -12%); the fresh-2023 results are entirely 2024.**

**Selection (pre-registered rule):** D12 score 1.00, D7 0.95 -> tie within 0.05 -> lower full MDD -> **D12**
(G 1.0, k = min(floor(E/25), floor(0.2N), floor(N/4))). Caveat: the kq rule is the "both legs floor(0.2N)"
convention that was also the best of the four conventions in step 1; the fixed-k rules (k6: 1.01, k8: 0.85)
show how much of the edge depends on the exact count. Its 2020-21 advantage comes from 3-name legs in 2020.

## Step 4 — stress of the selected D12 (s4_final.py, no new selection)

| run | SR | $200 -> | MDD | years |
|---|---|---|---|---|
| 2x cost (0.13%/side) | 1.10 | 798 | -30% | +84/+41/+5/-9/+63 |
| fresh $200 on 2021-01-01 | 1.04 | 592 | -31% | 2021 +62, 2022 +11, 2023 -4, 2024 +73 |
| fresh $200 on 2022-01-01 | 0.96 | 354 | -25% | 2022 +6, 2023 -5, 2024 +76 |
| lag 2 days | 0.22 | 222 | -54% | +36/-22/+2/-17/+25 |

Deflated Sharpe (lab.deflated_sharpe, trial SR dispersion 0.5): D12 SR 1.32, skew 0.72, kurt 13.2 -> **0.26 at n=855,
0.26 at n=892** (desk + this verifier). Hunter headline 1.34 -> 0.28; own-convention 1.20 -> 0.18. Not significant
after the desk's multiple testing.

## Step 5 — live-universe translation (s5_universe.py; translation check, not selection)

Live job cannot rank against 150 Binance coins. Ranking inside the 92-symbol HYPE list by 30d median quote volume:

| universe | names/yr 2020..2024 | full SR, $200 -> | fresh-2023 SR, $200 -> | lag1 |
|---|---|---|---|---|
| IS (Binance top-50 of 150, then HYPE) | 16/30/34/42/49 | 1.32, 1094 | 1.38, 350 | 1.00 |
| HYPE-list top 50 | 16/30/34/43/50 | 1.34, 1129 | 1.37, 347 | 1.02 |
| HYPE-list top 40 | 16/30/34/40/40 | 1.21, 949 | 1.07, 309 | 0.97 |

Top-40 was my a-priori spec guess (made from a wrong belief that IS held ~37 names); top-50 is adopted because its
name counts match the IS universe, not because of its Sharpe. Dropping names ranked 41-50 cuts 2024 from +78%
to +53%: part of the edge sits in the least liquid tail -> slippage/capacity risk above $10k.
Configs/runs by this verifier: step1 19 + step2 6 + step3 12 (x3 runs) + step4 4 + step5 2 = 43.
Desk total incl. this verifier ~898.

## Frozen $200 spec (proposed) — PKCC30-200

See the structured output `proposed_spec` (copied verbatim below).

```
PKCC30-200 v1 (frozen 2026-10-24 by verifier pkcc_impl)
VENUE   Hyperliquid USDC perps, start equity 200 USDC. Margin: CROSS, per-asset leverage set to 3x (headroom only;
        book gross is 1.0x equity so effective account leverage ~1.0). Assets that are isolated-only on HYPE: isolated 2x.
        No stop-loss / take-profit orders.
SYMBOLS Frozen list = the 92 rows of alpha_20261024/hype_map.csv with a HYPE name. No additions.
JOB     GitHub Actions cron daily 00:07 UTC (any start 00:05-01:00 UTC is fine). For each symbol pull HYPE
        candleSnapshot interval=1d, last 70 COMPLETED bars (bar open 00:00 UTC; drop the in-progress bar).
UNIVERSE at day t (bar that just closed): >=60 completed HYPE daily candles; day-t candle present;
        dqv = median over last 30 days (>=15 obs) of v*c; rank descending among eligible; keep rank <= 50.
SIGNAL  per day d: pk_d = (ln(h_d/l_d))^2/(4 ln2); cc_d = (ln(c_d/c_{d-1}))^2 (c_{d-1} = previous day's close; if
        missing, day d is excluded from BOTH sums). ratio = sqrt( mean_30d(pk) / mean_30d(cc) ) over joint obs of
        the last 30 days, require >=24 obs; drop if mean cc == 0 or ratio not finite.
        N = number of universe names with a valid ratio.
SIZE    E = account equity (USDC, marked) at run time. k = min(floor(E*1.0/2/12.5), floor(0.2*N), floor(N/4)).
        If k < 3 -> target flat for all.  (E=200 -> k=8 -> 16 positions of 12.50 USDC.)
SELECT  sort valid names by ratio ascending (ties: symbol A-Z). LONG the k lowest ratio, SHORT the k highest.
WEIGHTS each long +E/(2k), each short -E/(2k) notional; gross 1.0, dollar-neutral.
ORDERS  names leaving the book: reduce-only market close (any size). Others: delta = target - current notional;
        send market/IOC only if |delta| >= 10 USDC, else leave. Closes/reductions before opens. Rebalance daily.
COSTS   IS assumption 0.065%/side + hourly funding charged.
GUARDS  (operational, NOT backtested) halt + review if equity < 60% of peak (IS max DD -29%).
IS REF  2020-03..2024-12, $200, account_sim: SR 1.32-1.34, $200->$1094-1129, MDD -29%; fresh $200 2023-01-01: SR 1.37,
        $200->$347, MDD -24%; 2x cost 1.10; lag1 1.00; lag2 0.22; 2023 calendar year -2..-4%.
```

## Verdict (pkcc_impl)

**confirmed_with_concerns.** The hunter's numbers reproduce exactly and there is no lookahead (truncation test
bit-identical at 3 cuts). The headline 1.34 is not robust to equally valid readings of the rule (1.08–1.41 from
one name per leg) and gets +0.16 in 2022 from a stale-data artefact. A $200 book with daily rebalance and
k=floor(0.2N) (D12) holds up in the small-account sim: 1.32, lag1 1.00, 2x cost 1.10. But fixed-k books fall to
0.85–1.03, every config loses in 2023, lag2 kills it, and the deflated Sharpe at the desk's ~890 trials is
~0.26. P(forward Sharpe > 0.5) ~0.38. Forward-test it only as a small paper/$200 book, with the spec frozen above.
