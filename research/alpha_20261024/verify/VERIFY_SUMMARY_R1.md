# Round 1 verification summary — Funding-Crowding L/S (FCL), IS 2020-03 → 2024-12 only

The four verifier agents hit the session limit before writing their reports. Their result files (JSON) were
complete for most checks. The lead read those files and wrote this summary. Sources: `v1_reimpl_audit/*.json`,
`v2_overfit/s*.json`, `v3_hype_design/p*.json`, `v4_risk/t*.json`, `../final/is_check.json`.

## Status: alpha found, NOT deployable under the user's constraints

- The signal needs **Binance** funding history (geoblocked, and the user runs no VPS).
- The same signal built from **HYPE funding** loses money. On the 2023-06 → 2024-12 overlap, the HYPE-signal
  version has Sharpe −0.8 with HYPE prices and −1.5 with Binance prices. The Binance-signal version has −0.1 / +0.1
  on the same window.
- The rank correlation between Binance and HYPE 14d funding is only about 0.55.

## Checks that passed

| Check | Result |
|---|---|
| Independent reimplementation (own loop, rules text only) | Sharpe 1.49 vs lab 1.53 vs hunter 1.61. The gap comes from the tercile boundary rule. With q 0.30–0.35, Sharpe is 1.47–1.74 |
| Funding attribution vs raw Binance files (8 coins) | 0 mismatches. The settlement at the trade time is correctly not credited |
| Truncation test (delete future data, recompute) | identical up to every cut date → no lookahead |
| No-funding (price part only) | Sharpe 1.01 → not pure carry |
| Block bootstrap (20d) 90% CI of Sharpe | 0.81 – 2.35; P(SR<0.5) = 1.3% |
| Parameter surface 96 cells (L × q × top_k) | median 1.32, 85% of cells > 1, none < 0 |
| Survivorship variants (pre-2022 coins only / HYPE-listed only) | 1.52 – 1.67 |
| Data fixes (ICP funding hole, stale prices, BNB 0%-interest funding) | 1.63 |

## Checks that raise doubt

| Check | Result |
|---|---|
| Deflated Sharpe, n = 433 desk trials | 0.60 (with SD 0.5) / 0.29 (with the family's empirical SD) |
| Ex-2021 | Sharpe 1.09; 2023-24 alone 0.90 (CI −0.51 … 2.39) |
| 2023-24 without its 10 best days | Sharpe ≈ 0.00 |
| Concentration | 10 best days = 40% of PnL; the top 10 coins = 97% of PnL |
| Weak regime | alts ≪ BTC (BTC dominance): Sharpe 0.55 |
| Small books | 4–6 names/side: 2023-24 Sharpe −0.23 … 0.43, MaxDD −50 … −75% |

## Risk overlays tested in IS by v4

All were pre-registered.

- Skipping shorts in the top-10% 14d momentum: 1.61 → 1.87. Every sub-period improves.
- Short stop +40% (on 1h closes): 1.99, but 2022 falls from 1.78 to 1.13.

## Final frozen spec FCL v1.0, IS replay (`final/is_check.py`)

The spec is: HYPE-listed top-50, K = equity-limited (8/side at $200), momentum skip, stop +40%, HYPE costs.

- $200 replay from 2020-03: $868, Sharpe 1.64, MaxDD −27%, 2023-24 Sharpe 1.00.
- A fresh $200 start on 2023-01: $285 at the end of 2024.

The forward test (2025+) was **not** run for FCL at this point.
