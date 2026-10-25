# Changelog

## 2026-10-25 — infra v1.0.2 (aturan strategi tidak berubah)
- Feed BinPerp putus ≥ 14 hari: semua posisi paper ditutup, `control/bot.yaml` → `mode: off` (di-commit `save_state.sh`), alarm. Setelah dinyalakan lagi, hitungan putus mulai dari nol dan 15 hari terakhir diisi dari arsip.
- 2 tes baru (17 total).

## 2026-10-25 — infra v1.0.1 (aturan strategi tidak berubah)
- Gagal `predictedFundings` tidak lagi menghentikan mark, stop short, dan funding HYPE.
- Feed BinPerp putus ≥ 8 jam: alarm, run dihitung error, dan ledger diisi otomatis dari arsip premium index Binance (`fclbot/bootstrap.py`); celah diisi juga setelah feed pulih.
- Universe melewati koin yang candle terakhirnya basi (delisting HYPE); posisi yang hilang dari `allMids` 3 siklus ditutup di harga terakhir.
- > 10% candle gagal: rebalance ditunda ke siklus berikutnya, paling lama 4 jam setelah jadwal; jumlah candle gagal masuk pesan rebalance.
- Tanpa Telegram, pesan muncul sebagai anotasi `::notice` di run Actions.
- 6 tes baru (15 total).

## 2026-10-25 — v1.0 paper
- Bot paper FCL v1.0 di GitHub Actions (pola RNT): watcher 10 menit, watchdog, control, bootstrap.
- Funding Binance direkam dari `predictedFundings` HYPE; 16 hari pertama di-bootstrap dari arsip premium index Binance.
- Rebalance harian 08:20 UTC (15:20 WIB).
