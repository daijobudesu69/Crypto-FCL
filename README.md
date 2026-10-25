# Crypto FCL — Funding-Crowding Long/Short (paper test di HYPE)

> Dokumen lengkap (riset, keputusan, arsitektur, runbook): [docs/FCL_HANDOVER.md](docs/FCL_HANDOVER.md). Arsip riset: [research/](research/).

Bot **paper trading** untuk strategi FCL hasil riset Alpha Hunt 24 Okt 2026. Laporan lengkap:
`C:\Crypto data\backtest data and more\Alpha Hunt CHOP-30 20261024.md` (§6) dan
`Alpha Hunt - Perbandingan RNT, Fat Tail, Uji Hipotesis 20261024.md`.

- Modal paper **200 USD**, eksekusi (nanti) di **HYPE (Hyperliquid)**.
- **Tanpa VPS**: semua jalan di GitHub Actions (repo publik = menit Actions tidak terbatas).
- **Tanpa akses Binance**: funding Binance dibaca dari API HYPE (`predictedFundings`, venue `BinPerp`).
- Telegram, Google Sheets dan wallet HYPE **menyusul**. Sampai itu, pesan Telegram hanya dicetak di log Actions, dan catatan resmi ada di `state/`.

## Aturan strategi (FCL v1.0, sama dengan yang di-backtest)

| Item | Aturan |
|---|---|
| Koin | 91 perp HYPE di `coins.csv` (beku). Koin HYPE dikeluarkan karena tidak punya funding Binance di API |
| Universe | Sudah trading di HYPE ≥ 60 hari, top 50 menurut median volume USD 30 hari, cakupan data funding ≥ 70% |
| Sinyal | **F14** = funding Binance 14 hari (rata-rata per jam × 336) |
| Short | k koin F14 **tertinggi** (long ber-leverage yang ramai). Dilewati: koin dengan return 14 hari di 10% teratas (anti-squeeze) dan koin yang kena stop sejak rebalance sebelumnya |
| Long | k koin F14 **terendah** |
| k | `min(floor(N/3), max(4, floor(ekuitas / 25)))`. Di $200, k = 8, jadi 16 posisi × $12,50 |
| Bobot | Sama rata, +50% long / −50% short (dollar-neutral, gross 1×) |
| Stop | Short ditutup kalau harga ≥ entry × 1,40 (+0,5% slippage) |
| Order | Selisih < $10 tidak di-trade (kecuali buka/tutup). Biaya paper 0,065%/sisi + funding HYPE riil per jam |

Hasil riset (akun $200, biaya HYPE):

| Periode | Hasil | Sharpe | MaxDD |
|---|---|---|---|
| Backtest 2020-03 → 2024-12 | $200 → $868 | 1,64 | −27% |
| Forward 2025-01 → 2026-10 | $200 → $373 | 2,33 | −6,6% |

Periode forward sangat menguntungkan strategi ini (alt −57%). **Ekspektasi realistis: Sharpe 1–1,5, DD 15–30%.**

## Jadwal: rebalance 15:20 WIB (08:20 UTC)

Alasan tidak memakai 07:00 WIB (00:00 UTC):
1. **Sinyal paling segar.** Funding Binance settle pukul 00/08/16 UTC untuk koin 8 jam, dan tiap 4 jam untuk koin 4 jam. Pukul 08:20 UTC settlement 08:00 sudah masuk ledger, jadi tidak ada bagian sinyal yang masih "predicted".
2. **Menghindari jam 00:00 UTC.** Close candle harian adalah jam tersibuk: banyak bot (termasuk RNT dan RMF Anda) rebalance di menit yang sama, spread melebar, dan Telegram Anda paling ramai.
3. **Strategi ini lambat.** Di backtest, sinyal yang telat 1 hari tetap memberi Sharpe 1,65 (vs 1,61). Candle harian yang dipakai untuk momentum dan volume berumur 8 jam, dan itu tidak berpengaruh.
4. **Jam yang nyaman.** Sore WIB: Asia masih aktif, sebelum lonjakan volatilitas pembukaan bursa AS (20:30 WIB), dan Anda masih terjaga untuk mengecek.

## Cara kerja (tanpa VPS)

```
bot.yml (cron 7,37 * * * *) -> watcher hidup ~5,5 jam, siklus tiap 10 menit,
                               bangun juga 5 menit sebelum tiap settlement Binance (kelipatan 4 jam UTC)
  └─ run_cycle.py
       1. predictedFundings HYPE -> snapshot funding Binance -> state/funding_ledger.csv saat settlement lewat
       2. buku paper: mark harga mid, cek stop, funding HYPE per jam
       3. sekali per hari >= 08:20 UTC: candle harian HYPE -> universe, momentum, F14 -> rebalance paper
       4. outbox Telegram, log state/runs.csv
  └─ tools/save_state.sh  commit state/ ke repo
watchdog.yml   alarm + nyalakan watcher baru kalau state tidak diperbarui > 90 menit
control.yml    ganti mode (off / paper)
bootstrap.yml  isi ulang ledger funding dari arsip Binance kalau bot lama mati
ci.yml         tes offline
```

**Kenapa bisa tanpa VPS:** cron GitHub memang sering telat atau terlewat, jadi bot dirancang tetap berjalan walaupun ada siklus yang hilang.
- Watcher menyalakan penggantinya sendiri, dan watchdog menjadi cadangan.
- Snapshot funding yang terlewat cukup dilewati: F14 adalah rata-rata dari jam yang tercatat.
- Uji IS: sinyal dari 1 snapshot per hari tetap memberi Sharpe 1,71. Dengan 20–40% hari terlewat, Sharpe 1,24–1,35.

**Bootstrap:** arsip Binance tidak punya funding harian, jadi `tools/bootstrap_funding.py` merekonstruksi 16 hari terakhir dari arsip premium index Binance.
- Validasi September 2026: korelasi 0,94 dengan funding asli.
- Baris `bootstrap` otomatis tergantikan oleh rekaman bot (`predicted`). Setelah 14 hari, sinyal 100% dari rekaman bot.

## Perbedaan dengan backtest (diketahui, disengaja)

| Hal | Backtest | Bot |
|---|---|---|
| Ranking likuiditas | Top-50 dari semua perp Binance, lalu disaring yang ada di HYPE (rata-rata 48 koin) | Top-50 koin HYPE menurut volume HYPE |
| Funding | Rate settled Binance | Snapshot predicted ±5 menit sebelum settlement; 14 hari pertama sebagian rekonstruksi |
| Jam rebalance | 00:00 UTC | 08:20 UTC |
| Cek stop | Close 1 jam | Harga mid tiap ±10 menit (live nanti: order trigger di exchange) |
| Koin HYPE | Ikut | Tidak (tidak ada funding Binance) |

Dengan input yang sama, pilihan long/short bot cocok 94–96% dengan kode riset (`python tools/parity_check.py`, butuh data lake). Sisanya berasal dari definisi universe di atas.

## Perintah

```bash
python run_status.py
```

Ekuitas paper, posisi, dan ranking F14 saat ini (tidak mengubah state).

```bash
gh workflow run control.yml -f mode=off
```

```bash
gh workflow run bot.yml -f mode=loop
```

```bash
python -m pytest -q tests
```

## Evaluasi paper test

- **Minggu 1–2:** cek `state/runs.csv` (siklus jalan tiap ≤ 1 jam) dan `state/funding_ledger.csv` (ada baris `predicted` untuk tiap settlement 4/8 jam).
- **Minggu 4:** bandingkan rate `predicted` di ledger dengan funding settled di arsip bulanan Binance (data lake). Kalau korelasinya < 0,9, perbaiki perekaman sebelum live.
- **Bulan 2–3:** bandingkan hasil paper dengan replay riset (`fcl.run`) di periode yang sama. Selisih besar = ada masalah eksekusi.
- **Rem:** DD paper > 25% atau Sharpe rolling 90 hari < 0 → review sebelum lanjut.

## Menyusul

| Komponen | Yang dibutuhkan |
|---|---|
| Telegram | Secret repo `TELEGRAM_BOT_TOKEN` dan `TELEGRAM_CHAT_ID` (kodenya sudah siap, sama dengan RNT) |
| Google Sheets | Mirror log `state/*.csv` seperti RNT (`sheets.py`). Belum dipasang |
| Wallet HYPE | API wallet di subaccount khusus, mode `live/manage/flatten`, stop sebagai order trigger. Belum dipasang; mode tersebut saat ini diperlakukan sebagai `paper` |
