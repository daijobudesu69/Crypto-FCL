# CHOP-30 v1 — frozen spec (2026-10-24 10:07 UTC, written BEFORE any 2025+ data was opened)

Code: `final/strategies.py` (`chop_targets`, `replay`), hashes in `final/frozen_hashes.txt`
(strategies.py sha256 `1ac6fa10…`, lab.py `f262cf3e…`).

## Rules (HYPE perps, daily, GitHub Actions — no VPS, HYPE candles only)

| Item | Rule |
|---|---|
| Run | daily, about 00:05–01:00 UTC, on the last completed HYPE 1d candle (00:00 UTC) |
| Coin list | the 92 HYPE perps in `hype_map.csv` (frozen) |
| Universe | ≥ 60 days of HYPE trading (volume > 0), candle at t, top 50 by 30-day median daily USD volume (≥ 15 obs) |
| Signal | `ratio = sqrt( mean30[(ln H/L)² / (4 ln 2)] / mean30[(ln C_t/C_t-1)²] )`, ≥ 24 joint obs, drop if not finite |
| Meaning | ratio = Parkinson (high-low range) vol ÷ close-to-close vol. **High** = big intraday ranges that do not turn into close-to-close moves (choppy, wick-heavy coin). **Low** = closes move as much as the range (clean, directional coin) |
| Book | sort by ratio ascending (ties A–Z). **Long the k lowest, short the k highest** |
| k | `min(floor(E/25), floor(0.2·N), floor(N/4))`, E = equity, N = valid names. If k < 3, go flat. At $200: k = 8 → 16 positions × $12.50 |
| Weights | +E/(2k) per long, −E/(2k) per short. Gross 1.0×, dollar-neutral |
| Orders | close names that leave the book (reduce-only, any size). Others: trade only if \|target − current\| ≥ $10. Closes before opens |
| Rebalance | daily. Never trade on a signal older than 1 day (lag-2 IS Sharpe 0.22) |
| Margin | cross, dedicated subaccount, effective leverage ≈ 1× |
| Costs in tests | 0.065% per side (HYPE taker 0.045% + 0.02% slippage) + actual funding |

## IS evidence (2020-03 → 2024-12, Binance OHLC as HYPE proxy, HYPE-listed coins)

- Ideal book: Sharpe 1.37, MaxDD −27%.
- $200 account: $200 → $1,094, Sharpe 1.32, MaxDD −29%.
- Fresh $200 from 2023: $350, Sharpe 1.39.
- 2× cost: 1.10. Lag 1 day: 1.00. Lag 2 days: 0.22.
- Every year positive except 2023 (−4/−5%).

Doubts (verify_r2):
- Deflated Sharpe ≈ 0.26–0.28 at ~877 desk trials.
- Survivorship: young coins carry 44% of price PnL, and the long leg buys crash-bounces of coins that survived.
- Leg-count convention moves the Sharpe by ±0.3.
- Verifiers' P(forward Sharpe > 0.5) = 0.38–0.40.

## Forward-test protocol (fixed now)

- **Primary result:** HYPE-native replay (HYPE OHLC + HYPE funding + HYPE-volume universe), **$200 start on
  2025-01-01**, to the last data day (2026-10-02). Warm-up uses 2024 data.
- **Secondary results:**
  - Binance-data replay, $200.
  - Ideal fractional book.
  - BTC buy & hold $200 and an equal-weight HYPE-universe long as benchmarks.
- **Verdict rule (decided before looking):**
  - Forward Sharpe ≥ 0.5 **and** MaxDD better than −35% → "edge holds (weak confidence)".
  - 0 < Sharpe < 0.5 → "inconclusive".
  - Sharpe ≤ 0 → "failed: no tradable alpha shown".
- **Runner-ups**, also frozen now and reported alongside without changing the primary choice:
  - DISP7-200 (dispersion-timed alts vs BTC; `disp_targets`).
  - FCL v1.0 (funding-crowding; not deployable, Binance funding signal; `fcl.py`).
- No parameter of any strategy may be changed after the forward data is opened. The forward run is executed **once**.
