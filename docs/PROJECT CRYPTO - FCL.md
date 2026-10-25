# FCL — dokumen lengkap: riset, keputusan, bot, dan runbook

> **Untuk Claude di sesi berikutnya:** baca file ini dulu sebelum menyentuh repo Crypto-FCL. Semua yang dikerjakan di sesi 24–25 Okt 2026 ada di sini: cara strategi ditemukan, angka-angkanya, alasan setiap keputusan, arsitektur bot, dan cara menangani masalah teknis.
>
> Bahasa user: Indonesia, jawaban singkat dengan angka. Sebut Hyperliquid sebagai **"HYPE"** (bukan "HL"); tokennya disebut "koin HYPE".

---

## 0. Status singkat (per 25 Okt 2026 02:07 UTC)

| Item | Status |
|---|---|
| Strategi | **FCL v1.0: Funding-Crowding Long/Short**. Short koin dengan funding Binance 14 hari tertinggi, long yang terendah |
| Bot | **Paper trading berjalan** di GitHub Actions sejak **24 Okt 2026 18:11 UTC**, modal paper **$200** |
| Repo | https://github.com/daijobudesu69/Crypto-FCL (public), lokal `C:\Crypto data\Crypto-FCL` |
| Rebalance | Harian **08:20 UTC = 15:20 WIB** |
| Sumber sinyal | Funding Binance via API HYPE `predictedFundings` (venue `BinPerp`). Tanpa VPS, tanpa akses Binance |
| Terverifikasi live | Rekaman funding `predicted` masuk untuk settlement 20:00 dan 00:00 UTC (136 baris). Watcher bersambung. Equity tercatat per jam |
| Belum ada | Telegram (secret belum diisi), Google Sheets, wallet HYPE / mode live |
| Ekspektasi jujur | Sharpe ±1–1,5, DD 15–30%. Forward Sharpe 2,3 terjadi di periode yang sangat menguntungkan |

---

## 1. Permintaan user dan kronologi keputusan

### Permintaan awal (24 Okt 2026)
1. Pakai data crypto di data lake.
2. Buat **satu** strategi dan temukan alpha-nya, dikerjakan seperti quant profesional.
3. Strategi lama di folder (CLR, RMF, RNT, MEX, stat-arb) hanya sebagai referensi.
4. Backtest pakai **2020–2024**, forward test **2025 ke atas**.
5. Jujur: kalau tidak menemukan alpha, beri disclaimer.
6. Modal awal **200 USD**.
7. Laporan dalam file MD.
8. Boleh menyarankan strategi lain yang menarik.

### Perubahan syarat selama sesi

| Waktu | Kata user | Dampak |
|---|---|---|
| Saat ronde 1 berjalan | "kita akan running dengan DEX hyperliquid" | Universe = 92 koin Binance yang juga ada di HYPE. Fee HYPE (taker 0,045%), min order $10, funding HYPE per jam |
| Setelah ronde 1 | "dont use strategy that make me use VPS, like funding, etc. ... binance is have geoblocks" | FCL (sinyal funding Binance) dianggap tidak bisa dideploy. Ronde 2 hanya memakai OHLCV HYPE. Strategi utama di laporan pertama menjadi **CHOP-30** |
| Setelah laporan | Bertanya bagaimana FCL bisa 100% tanpa VPS, dan HYPE atau Binance | Ditemukan API HYPE `predictedFundings` yang memuat funding Binance. FCL bisa jalan tanpa VPS di HYPE |
| 25 Okt | "siapkan semua infrastrukturnya dan mulai jalankan paper test secepatnya ... gunakan RNT as reference ... cron daily tidak harus jam 7 pagi WIB" | Bot Crypto-FCL dibuat dan dijalankan. Rebalance dipindah ke 15:20 WIB |

Catatan memori user yang relevan:
- "No OOS without asking": jangan menjalankan OOS/holdout baru tanpa diminta. Forward 2025+ di sesi ini **diminta** user.
- User tidak mau biaya VPS, dan Binance diblokir dari jaringannya dan dari runner GitHub di AS.

---

## 2. Data dan harness riset

### Data lake (`C:\Crypto data\backtest data and more\data`)
- **Binance USDT-M**, 150 koin top yang masih hidup (ada survivorship bias, disetujui user), 2020 → 2026-10-02:
  - kline 1m–1d,
  - funding (sampai 2026-09-30, dari arsip bulanan),
  - metrics (OI, L/S ratio, taker; per 5 menit),
  - premium index 1 jam,
  - spot.
- **HYPE:** candle 1d (sebelum listing berisi harga oracle hasil backfill dengan **volume 0**), 4h sejak 2024-06, funding per jam sejak 2023-05.
- Jaringan user: `fapi.binance.com`, Bybit, Deribit dan Coinbase **diblokir**. `data.binance.vision`, `data-api.binance.vision` dan `api.hyperliquid.xyz` bisa diakses.

### Harness `alpha_20261024/` (salinan arsip ada di `research/alpha_20261024/` repo ini)

| File | Fungsi |
|---|---|
| `build_cache.py` | Panel lebar float32 (index = waktu open bar UTC, kolom = simbol) untuk 1d/4h/1h → `cache/` (313 MB, tidak masuk repo) |
| `build_cache_hype.py` | Panel HYPE harian (`hype_1d`) dengan nama kolom Binance |
| `lab.py` | Loader + **split guard** + backtest + metrik |
| `account_sim.py` | Replay akun kecil ($200, minimum order $10) |
| `hype_map.csv` | Peta simbol Binance → nama HYPE (92 koin) |
| `fcl.py` | Implementasi FCL v1.0 (replay hari per hari, stop pakai close 1 jam) |
| `final/strategies.py` | CHOP-30 dan DISP7 + replay akun |

Aturan di `lab.py`:
- **Split guard:** semua loader memotong data di 2025-01-01, kecuali env `ALPHA_FORWARD=1`. Agen riset dilarang menyalakannya.
- **Konvensi waktu:** nilai panel di baris t diketahui saat close bar t. Bobot W[t] di-trade di close t dan mendapat return bar t+1, plus funding `fund[t+1]` (settlement dalam rentang (close t, close t+1]).
- **Biaya default:** 0,07%/sisi (Binance) dan `HYPE_COST` 0,065%/sisi (taker 0,045% + slip 0,02%), dihitung atas turnover yang disesuaikan drift, ditambah funding riil.
- **Metrik:** CAGR, Sharpe, Sortino, MaxDD, t Newey-West, skew/kurtosis, turnover, drag biaya, funding per tahun, dan deflated Sharpe (Bailey & López de Prado).
- `universe(tf, top_k)`: likuiditas point-in-time (median 30 hari volume USD, ≥ 60 hari histori). `hype_universe()` = top-k **dari semua 150 koin Binance**, lalu disaring yang ada di HYPE (rata-rata 48 koin di 2025+).

### Cara menjalankan ulang riset
Kode riset harus dijalankan dari lokasi aslinya `C:\Crypto data\backtest data and more\alpha_20261024\`, karena butuh `../data` dan `cache/`. Salinan di repo ini hanya arsip (tanpa cache/parquet besar).
- Cache dibangun ulang dengan `python build_cache.py`, lalu `python build_cache_hype.py`. Butuh ±15 menit.

---

## 3. Metodologi riset ("pro quant")

1. **Hipotesis dulu, dengan alasan ekonomi:** siapa pihak lawannya, dan kenapa efeknya bertahan.
2. **Grid ditulis sebelum dijalankan (pre-registered)**, maksimal ±40 config per keluarga. Semua trial dihitung: ±433 di ronde 1, ±422 di ronde 2, ±60 oleh verifier. Totalnya **±900**, dipakai untuk deflated Sharpe.
3. **Uji stres wajib:** 2× biaya, telat 1 bar (`extra_lag=1`), plateau parameter tetangga, Sharpe per sub-periode (2020–21, 2022, 2023–24), dekomposisi leg long/short, korelasi dengan BTC dan momentum XS 14 hari, serta versi akun $200.
4. **Verifikasi adversarial:** agen lain mencoba menjatuhkan kandidat lewat re-implementasi dari teks aturan saja, audit lookahead (uji pemotongan data), kritik overfit (bootstrap, DSR, survivorship), dan review risiko.
5. **Freeze lalu forward sekali:** spec + hash kode + aturan lulus/gagal ditulis **sebelum** data 2025+ dibuka, lalu forward dijalankan sekali untuk semua kandidat.
6. Riset dikerjakan oleh "hunter" paralel lewat Workflow tool (5 agen per ronde × 2 keluarga hipotesis), lalu verifier 4 agen per ronde.

**Insiden teknis:** workflow verifikasi ronde 1 terkena batas sesi; keempat agen mati sebelum menulis laporan. Hasil JSON antara mereka dibaca langsung, lalu diringkas di `research/alpha_20261024/verify/VERIFY_SUMMARY_R1.md`. Di ronde 2, agen diminta menulis hasil per langkah.

---

## 4. Ronde 1: semua data diizinkan (12 keluarga)

| # | Hipotesis | Hasil IS 2020–2024 | Verdict |
|---|---|---|---|
| 1 | **Funding-crowding cross-sectional** | Tercile L14: Sharpe 1,61, t 3,37. Plateau rank-weighted 1,47–1,71. 2× biaya 1,40, lag 1 1,65 | ✅ **kandidat utama** |
| 2 | Cash-and-carry spot/perp | BTC+ETH selalu-on: 16,7% CAGR atas notional, Sharpe 9,5 (2022 hanya +2,4%). Versi selektif high-funding malah rugi | premi pasar, bukan alpha |
| 3 | Timing settlement funding (1h) | Semua efek < 10 bp. Efek +45 bp hanya muncul dengan informasi yang belum diketahui saat entry | ❌ |
| 4 | Positioning top trader vs retail (L/S) | Arah "top trader informed" ditolak. Kontrarian crowding 1,33–1,52, tapi efektif hanya 2023–24 | lemah; data tidak bisa diambil live |
| 5 | Dinamika OI | Terbaik 0,82, gagal di 2× biaya | ❌ |
| 6 | Basis spot vs perp 7 hari | 1,21–1,35, semua sub-periode positif | butuh futures Binance live |
| 7 | Taker flow / lonjakan volume | Momentum terselubung (korelasi 0,57) | ❌ |
| 8 | Short-term reversal 1–5 hari / 4 jam | Yang ada justru continuation. Reversal 4 jam gross Sharpe ±1, tapi turnover 2.200–3.400×/tahun | ❌ (hanya bisa dengan maker) |
| 9 | BAB / IVOL / MAX / skew | Arah buku teks gagal di 28/28 config | ❌ |
| 10 | Seasonality jam/hari | Tidak persisten | ❌ |
| 11 | BTC → alt lead-lag | Tanda terbalik | ❌ |
| 12 | Vol-managed basket | Alpha t < 1,1 | ❌ |

### Temuan kunci keluarga funding (hunter h1)
- **Bukan carry murni.** Tanpa funding, Sharpe dari harga saja ±1,0–1,3. Carry menambah ±10%/tahun, hampir semuanya dari leg short.
- **Edge ada di leg short.** Leg short vs EW: Sharpe 2,2 (positif tiap tahun). Leg long vs EW: 0,6 (±0 di 2020/2023/2024).
- **Faktor baru.** Korelasi dengan momentum XS 14 hari +0,09, dengan BTC −0,07.
- **Kelemahan:**
  - 2021 menyumbang lebih dari separuh log-return.
  - Top-5 koin menyumbang 53% PnL.
  - Rolling 365 hari pernah −1,45 (Juni 2024).
- **Delta funding** (perubahan funding) tidak punya edge.

---

## 5. Verifikasi FCL (IS saja)

Sumber: `research/alpha_20261024/verify/VERIFY_SUMMARY_R1.md` + JSON di `verify/v1..v4`.

### Lolos
- **Re-implementasi independen** dari teks aturan: Sharpe 1,49, vs lab 1,53, vs hunter 1,61. Selisihnya dari batas tercile (q 0,30–0,35 → 1,47–1,74).
- **Atribusi funding** dicek terhadap file mentah Binance (8 koin): 0 mismatch. Settlement tepat di waktu trade **tidak** dikreditkan.
- **Uji pemotongan data** di 5 tanggal: hasil sebelum tanggal potong identik → **tidak ada lookahead**.
- **Bootstrap** (blok 20 hari) CI 90%: 0,81–2,35; P(Sharpe < 0,5) = 1,3%.
- **Permukaan 96 sel** (L × q × top_k): median 1,32, 85% sel > 1, tidak ada yang < 0.
- **Varian survivorship** (hanya koin listing < 2022 / hanya koin HYPE): 1,52–1,67.
- **Perbaikan data** (lubang funding ICP, harga basi, BNB dengan funding 0%): 1,63.

### Meragukan
- **DSR** dengan n = 433: 0,60 (SD 0,5) / 0,29 (SD empiris keluarga).
- Ex-2021: 1,09. 2023–24: 0,90 (CI −0,51..2,39). 2023–24 tanpa 10 hari terbaik: ±0.
- 10 hari terbaik = 40% PnL.
- Rezim lemah: alts ≪ BTC (Sharpe 0,55).
- **Sinyal dari funding HYPE GAGAL:** overlap 2023-06 → 2024-12, Sharpe −0,8 (harga HYPE) / −1,5. Korelasi ranking F14 HYPE vs Binance hanya 0,55. **Sinyal harus dari funding Binance.**
- **Buku kecil merusak edge:** 4–6 nama per sisi → Sharpe 2023–24 −0,23..0,43, MaxDD −50..−75%. 8 per sisi → DD −40%. Tercile (±23) terbaik. Karena itu **k diskalakan dengan ekuitas**.

### Overlay risiko (pre-registered oleh verifier v4)

| Overlay | Sharpe IS | Keterangan |
|---|---|---|
| Tanpa overlay | 1,61 | |
| Skip short dengan momentum 14 hari di top 10% | 1,87 | Semua sub-periode membaik → **dipakai** |
| Stop short +25% | 2,09 | 2022 turun |
| Stop short +40% | 1,99 | 2022 1,78 → 1,13; **dipakai sebagai stop darurat** |

### Ablasi FCL v1.0 final (`final/is_check.json`, HYPE-listed top-50, biaya HYPE, 2020-03 → 2024-12)

| Versi | $200 → | Sharpe | MaxDD | Sub 20-21 / 22 / 23-24 |
|---|---|---|---|---|
| Tercile tanpa overlay | $787 | 1,38 | −31% | 1,78 / 1,56 / 0,86 |
| Tercile FCL v1 | $927 | 1,62 | −26% | 2,14 / 1,91 / 0,89 |
| Tercile FCL v1, 2× biaya | $736 | 1,39 | −29% | 1,96 / 1,60 / 0,58 |
| Akun $200 tanpa overlay | $755 | 1,37 | −31% | 1,81 / 1,19 / 0,96 |
| Akun $200, momentum saja | $1.113 | 1,78 | −26% | 2,41 / 1,84 / 0,99 |
| Akun $200, stop saja | $756 | 1,43 | −29% | 1,87 / 1,11 / 1,06 |
| **Akun $200 FCL v1** | **$868** | **1,64** | **−27%** | 2,23 / 1,47 / 1,00 |
| Akun $200 FCL v1, mulai 2023 | $285 | 0,99 | −32% | – / – / 0,99 |

> Catatan: versi "momentum saja" lebih tinggi daripada v1 (yang memakai stop). Stop dipertahankan sebagai perlindungan ekor squeeze di buku kecil, bukan untuk mengejar Sharpe.

---

## 6. Ronde 2: hanya OHLCV HYPE (setelah syarat "tanpa VPS")

| # | Hipotesis | Hasil | Verdict |
|---|---|---|---|
| 13 | Seasonality XS slot 4 jam / weekday | Slot 4 jam: **gross nyata** (IC t 8,4; gross Sharpe 3,44), tapi break-even 2,6 bp/sisi vs biaya 6,5 bp → net −5 | ❌ (butuh maker + VPS) |
| 14 | Turn-of-month / akhir bulan / Senin | 0 dari 24 lolos validasi 2023–24 | ❌ |
| 15 | 52-week high / anchoring | Ternyata momentum (alpha t < 2) | ❌ |
| 16 | Residual momentum / frog-in-the-pan | ≤ 0,71, 2022 negatif | ❌ |
| 17 | Bentuk candle (CLV, wick) | Momentum terselubung | ❌ |
| 18 | Ukuran trade / FOMO retail | Ditolak | ❌ |
| 19 | Downside beta / coskew | BAB terselubung | ❌ |
| 20 | **Rasio vol Parkinson/close (→ CHOP-30)** | 1,34; 2× biaya 1,09; lag 1 0,92; lag 2 0,17 | ✅ dipilih untuk laporan |
| 21 | Umur listing | Rusak oleh survivorship | ❌ |
| 22 | **Dispersion timing alt vs BTC (DISP7)** | 1,38; plateau 1,27–1,70; lag 3 1,59 | ✅ runner-up |

Verifier ronde 2 (`research/alpha_20261024/verify_r2/verify_r2_results.json`):
- Keduanya "confirmed_with_concerns".
- DSR ±0,26–0,31 di n ±855; P(forward Sharpe > 0,5) 35–40%.
- **CHOP-30:** rapuh terhadap aturan jumlah koin per sisi (±0,3), sinyalnya cepat, dan 44% PnL harga datang dari koin muda.
- **DISP7:** 69% PnL berasal dari 3 episode, dan 2024 negatif.

Alasan CHOP-30 dipilih (berdasarkan IS saja): versi $200-nya lebih kuat (1,32 vs 0,93), dan jumlah bet independennya lebih banyak.

---

## 7. Forward test 2025-01-01 → 2026-10-02 (sekali jalan)

Protokol ada di `research/alpha_20261024/SPEC_CHOP30_v1.md`; hash kode di `final/frozen_hashes.txt` dan dicek sebelum run. Aturan verdict yang ditetapkan sebelumnya:
- Sharpe ≥ 0,5 dan DD lebih baik dari −35% → "bertahan".
- 0 < Sharpe < 0,5 → "inconclusive".
- Sharpe ≤ 0 → "gagal".

| Strategi ($200) | Akhir | Return | Sharpe | MaxDD | Catatan |
|---|---|---|---|---|---|
| **FCL, data HYPE** | **$373** | +86,6% | **2,33** (t 3,15) | **−6,6%** | 17/22 bulan positif, bulan terburuk −2,9%, funding diterima +$40,7, fee $25,1, 16 stop |
| FCL, data Binance | $382 | +90,8% | 2,40 | −7,4% | |
| CHOP-30, data HYPE | $221 | +10,5% | 0,38 (t 0,44) | −20% | **inconclusive**; tanpa Apr-2025 (+29%) hasilnya −14% |
| CHOP-30, data Binance | $273 | +36,5% | 0,96 | −20% | gap karena wick HYPE di data H/L (lihat bawah) |
| DISP7, HYPE / Binance | $195 / $124 | −2% / −38% | 0,10 / −0,77 | −32 / −53% | **gagal** |
| BTC buy & hold | $180 | −9,8% | 0,08 | −53% | |
| Alt EW long | $85 | −57% | −0,23 | −79% | |

Return bulanan FCL (HYPE), 2025-01 → 2026-10:

> +5,6, +13,5, +0,2, +6,8, +0,6, +1,6, +5,2, −2,2, +3,5, +10,3, +1,4, −2,8, +2,9, −0,5, +4,6, −2,9, +4,7, +3,4, −0,3, +2,6, +6,0, +0,6 (%)

Diagnosis gap CHOP-30 (post-hoc, tidak mengubah spec):

| Sinyal | Universe | Eksekusi | Sharpe |
|---|---|---|---|
| Binance | Binance | Binance | 0,96 |
| Binance | Binance | HYPE | 0,97 |
| Binance | HYPE | HYPE | 0,91 |
| **HYPE** | HYPE | HYPE | **0,38** |

Penyebabnya wick khas HYPE di data high/low: korelasi ranking sinyal 0,97, tapi 8 koin paling ekstrem di tiap sisi berbeda.

---

## 8. Analisis lanjutan (laporan `research/Alpha Hunt - Perbandingan RNT, Fat Tail, Uji Hipotesis 20261024.md`)

### Perbandingan dengan RNT v1.1
Sumber RNT: `Crypto-RS-Neutral & Trend/research/audit_response/v11_*_daily.csv`. Catatan: OOS RNT **tidak murni** (v1.1 dibuat setelah OOS v1.0 dilihat).

| | FCL | CHOP-30 | RNT | BTC |
|---|---|---|---|---|
| IS $200 → / Sharpe | $868 / 1,64 | $1.094 / 1,32 | $3.360 / 2,10 | $2.193 / 1,09 |
| Forward $200 → / Sharpe | $373 / 2,33 | $221 / 0,38 | $440 / 1,69 | $181 / 0,08 |
| Forward MaxDD / vol | −6,6% / 16% | −20% / 21% | −16% / 29% | −53% / 44% |

- Korelasi harian FCL–RNT: IS 0,08, forward 0,10.
- Gabungan 50/50 FCL + RNT: IS Sharpe 2,62 (DD −11%), forward **2,49 (DD −8,8%)**.

### Fat tail (2020-03 → 2026-10)

| | Excess kurtosis | Skew | Hari > 4σ (normal: 0,15) | Hari terburuk | CVaR 99% |
|---|---|---|---|---|---|
| FCL | +6,1 | +0,92 | 13 | −4,7% | 3,0% |
| CHOP-30 | +10,4 | +0,74 | 25 | −9,8% | 5,3% |
| RNT | +9,1 | +1,42 | 13 | −7,7% | 4,7% |
| BTC | +13,9 | −0,58 | 12 | −40% | 11,9% |

Jarque-Bera p < 0,001 untuk semua. Ekor kiri FCL paling tipis.

### 4 uji hipotesis (forward)

| Uji | FCL | CHOP-30 | RNT | BTC |
|---|---|---|---|---|
| 1. t Newey-West | 3,15 ✅ | 0,44 ❌ | 2,08 ✅ | 0,12 ❌ |
| 2. Bootstrap CI 90% / P(Sharpe ≤ 0) | [1,28; 3,35] / 0% ✅ | [−1,00; 1,72] / 36% ❌ | [0,45; 2,80] / 1% ✅ | 46% ❌ |
| 3. Permutasi label koin (p) | < 0,003 ✅ | 0,057 ❌ (versi Binance) | n/a | n/a |
| 4. DSR @ N = 900 | 0,83 ✅ | 0,05 ❌ | 0,54 | 0,02 |

Permutasi versi pertama (koin acak diacak ulang tiap hari) ternyata **tidak adil**: turnovernya jauh lebih besar, sehingga null-nya terlalu mudah dikalahkan. Versi yang dipakai mengacak pasangan sinyal→koin sekali per run (`stats/perm_fair.py`), jadi turnover dan biayanya sama.

### Uji survivorship dengan koin delisting (25 Okt 2026)

Sumber: repo user **Crypto-Funding-Strategy** (https://github.com/daijobudesu69/Crypto-Funding-Strategy). Salinan lokal ada di `C:\Crypto data 2`, dan panelnya `C:\Crypto data 2\data\panel.parquet`:
- 776 perp USDT Binance, **termasuk yang sudah delisting**;
- harian 2024-08 → 2026-07;
- kolom `close_t`, `adv_usd_30`, `funding_24h_sum` (settled Binance).

Skrip: `research/alpha_20261024/stats/fcl_survivorship_cds2.py`. FCL inti (tanpa stop), universe top-50 menurut ADV 30 hari point-in-time, biaya 0,065%/sisi + funding.

| Universe kandidat | Tercile Sharpe / MaxDD | k = 8 Sharpe / MaxDD |
|---|---|---|
| **Semua 776 termasuk delisting** | **1,60 / −17%** | **1,33 / −27%** |
| Semua tanpa koin HYPE | 1,68 / −17% | 1,20 / −28% |
| Hanya 150 koin yang hidup sekarang (seperti data lake) | 2,25 / −14% | 2,01 / −23% |
| Hanya koin HYPE (92) | 2,39 / −9% | 1,95 / −16% (2026 YTD −7%) |

Kesimpulan:
- **Edge tetap ada** dengan koin mati di dalam data.
- Koin yang kemudian delisting hanya ±0,5% dari posisi, karena universe top-50 likuid jarang memuatnya. Jadi survivorship bias *langsung* untuk FCL kecil.
- Turunnya Sharpe (2,25 → 1,60) lebih banyak karena universe yang lebih luas (koin hidup di luar top-150 ikut bersaing masuk top-50) daripada karena koin mati.
- Ekspektasi live yang realistis tetap Sharpe ±1,2–1,6.

Catatan lain dari repo tersebut:
- **Tidak bertentangan dengan FCL.** Temuan mereka "funding 24 jam tinggi → return 48 jam lebih tinggi" adalah efek jangka pendek. Di riset kita, FCL dengan window 1–3 hari juga lemah; edge FCL ada di funding **persisten 14 hari**.
- Logger mereka (`oi_logs/`, Gate.io `contract_stats` per jam sejak 2026-08-23, termasuk `last_funding_rate`) adalah funding **Gate.io**, bukan Binance. Data ini belum diuji sebagai sinyal; ingat bahwa funding HYPE sebagai sinyal gagal. Potensinya hanya sebagai sumber cadangan kalau `predictedFundings` HYPE berhenti, dan harus divalidasi dulu.

---

## 9. Kelayakan tanpa VPS dan pilihan venue

- **Sumber data:** `POST https://api.hyperliquid.xyz/info {"type":"predictedFundings"}` → `[[coin, [["BinPerp", {fundingRate, nextFundingTime, fundingIntervalHours}], ["HlPerp", ...], ["BybitPerp", ...]]], ...]`. Ada 234 koin.
  - Untuk 91 koin kita, interval Binance 4 jam (45 koin) atau 8 jam (46 koin).
  - Koin **HYPE tidak punya BinPerp**, jadi dikeluarkan.
  - Info sebuah venue bisa `null`; kode sudah menanganinya.
- **Uji 1 snapshot per hari** (IS, `stats/fcl_daily_snapshot_test.py`): sinyal hanya dari settlement 00:00 × jumlah settlement per hari.

  | Versi | Hasil $200 | Sharpe |
  |---|---|---|
  | Semua settlement | $868 | 1,64 |
  | 1 snapshot/hari | $905 | 1,71 |
  | 20% hari terlewat | $576 | 1,24 |
  | 40% hari terlewat | $651 | 1,35 |

  Korelasi ranking F14 asli vs proxy: 0,937. Kesimpulan: **cron yang tidak andal tetap aman.**
- **Bootstrap:** arsip Binance hanya menyediakan funding **bulanan**; harian tidak ada (404). Premium index 1 jam harian tersedia (`/data/futures/um/daily/premiumIndexKlines/{SYM}/1h/{SYM}-1h-{date}.zip`, terbit ±H+1).
  - Rekonstruksi: `F = P + clamp(I − P, ±0,05%)`, P = rata-rata premium dalam interval, I = 0,01% per 8 jam.
  - Validasi September 2026: korelasi Pearson 0,94, ranking 0,77.
- **Venue: HYPE.** Binance kena geoblock dari jaringan user dan dari runner GitHub di AS, jadi bot Binance justru butuh VPS. Hasil forward FCL hampir sama ($373 di HYPE vs $382 di Binance).

---

## 10. Bot Crypto-FCL

### Struktur repo

```
config.yaml            parameter strategi/paper/jadwal (mengubah angka = strategi baru)
coins.csv              91 koin beku: binance -> hype (tanpa HYPE)
control/bot.yaml       mode: off | paper   (live/manage/flatten -> diperlakukan paper + peringatan)
run_cycle.py           satu siklus (dipanggil watcher)
run_watchdog.py        alarm + nyalakan watcher kalau runs.csv basi > 90 menit (salinan RNT)
run_status.py          ringkasan ekuitas, posisi, ranking F14 (tidak mengubah state)
fclbot/
  config.py            baca config.yaml + coins.csv
  hype.py              klien /info HYPE (salinan RNT + predicted_fundings), token bucket 1000 weight/menit
  ledger.py            rekam snapshot -> funding_ledger.csv, F14, pangkas 40 hari
  strategy.py          universe, sinyal, pemilihan long/short, k
  paper.py             buku paper: mark, funding HYPE, stop, rebalance
  jobs.py              orkestrasi siklus + pesan Telegram harian
  store.py             JSON atomik + CSV append (salinan RNT, tanpa Sheets)
  notify.py            Telegram + outbox (salinan RNT)
  control.py           mode
tools/
  bootstrap_funding.py isi ledger dari arsip premium index Binance
  parity_check.py      pilihan bot vs fcl.py riset (butuh data lake)
  refresh_state.sh, save_state.sh, save_control.sh, set_control.py   (pola RNT)
.github/workflows/
  bot.yml        cron "7,37 * * * *": watcher ~5,5 jam, siklus 10 menit, menyalakan penggantinya sendiri
  watchdog.yml   cron "13,28,43,58 * * * *"
  control.yml    manual: mode
  bootstrap.yml  manual: isi ulang ledger
  ci.yml         pytest saat push (kecuali state/, md, research/)
tests/           9 tes offline (FakeInfo)
research/        arsip riset (tanpa cache/parquet besar) + 2 laporan
docs/            dokumen ini
state/           ditulis bot (lihat state/README.md)
```

### Satu siklus (`jobs.run`)
1. Mode `off` → tidak melakukan apa-apa.
2. Panggil `predicted_fundings()`, filter ke 91 koin, lalu `ledger.record()`:
   - Snapshot per koin disimpan di `funding_pending.json`.
   - Begitu `now ≥ next_ms`, atau API sudah menunjukkan settlement berikutnya, snapshot terakhir ditulis ke `funding_ledger.csv` dengan `source=predicted`.
3. Kalau ada posisi: `all_mids()` → `paper.mark()` → `paper.stops()` (short dengan mid ≥ entry × 1,40 ditutup di mid × 1,005 + fee, lalu dilarang 1 rebalance) → `paper.funding()` (fundingHistory HYPE per jam untuk koin yang dipegang).
4. Kalau `now ≥ 08:20 UTC` dan hari ini belum rebalance:
   - Ambil candle 1d HYPE 80 hari untuk 91 koin (±2 menit).
   - Hitung `ledger.f14()` dan `strategy.targets()`.
   - Mark, lalu `paper.rebalance()`. Larangan stop dikosongkan.
   - Tulis targets.csv dan equity (rebalance), pangkas ledger, kirim pesan Telegram ringkasan.
5. Equity `mark` dicatat per jam. Hasil disimpan ke `paper.json`.

### Detail sinyal di bot
- `F14 = (jumlah rate / jumlah jam tercakup) × 336`. Tiap baris ledger mencakup `interval_h` jam. Untuk satu (koin, settlement), baris `predicted` mengalahkan `bootstrap`.
- Universe:
  - coin harus punya ≥ 60 candle harian HYPE dengan volume > 0, dan candle terakhir dengan volume > 0;
  - median 30 hari (volume × close) dengan ≥ 15 observasi;
  - top 50, lalu disaring cakupan funding ≥ 70% dan F14 ada;
  - N < 12 → flat.
- k = min(N // 3, max(4, floor(ekuitas × 1,0 / 2 / 12,5))).
- Momentum 14 hari dari close HYPE; short dilewati kalau rank pct > 0,90.
- Tie-break berdasarkan nama koin.

### Jadwal: kenapa 08:20 UTC (15:20 WIB)
1. Settlement Binance 08:00 sudah masuk ledger (paling segar, tidak ada bagian yang masih "predicted").
2. Menghindari 00:00 UTC (close harian): RNT/RMF dan banyak bot rebalance di menit yang sama, spread melebar, Telegram user paling ramai.
3. Strategi lambat: lag 1 hari di IS tetap memberi Sharpe 1,65.
4. Sore WIB user bisa memantau, dan waktunya sebelum volatilitas pembukaan bursa AS (20:30 WIB).

Watcher juga bangun 5 menit sebelum tiap settlement kelipatan 4 jam (00/04/08/12/16/20 UTC) supaya snapshot hampir final.

### Perbedaan bot vs backtest (disengaja, sudah diukur)

| Hal | Backtest (`fcl.py`) | Bot |
|---|---|---|
| Universe | Top-50 dari 150 koin Binance ∩ HYPE (rata-rata 48) | Top-50 koin HYPE menurut volume HYPE |
| Funding | Settled Binance | Predicted ±5 menit sebelum settlement; 14 hari pertama sebagian rekonstruksi |
| Jam | 00:00 UTC | 08:20 UTC |
| Stop | Close 1 jam Binance | Mid tiap ±10 menit |
| Larangan setelah stop | 1 rebalance (perilaku kode; teks spec bilang "sampai sinyal lepas") | 1 rebalance (mengikuti kode, karena hasil riset berasal dari kode) |
| Koin HYPE | Ikut | Keluar |

Paritas pilihan (`tools/parity_check.py`, 25 hari acak 2025–26, $200, input Binance):
- Long 186/200, short 180/200.
- Dengan HYPE disertakan: 191/200 dan 188/200.
- Sisa selisih berasal dari definisi universe.

### Run pertama (24 Okt 2026 18:11 UTC, rebalance pertama)

| Sisi | Koin |
|---|---|
| Long | DOT, TIA, JUP, ARB, NIL, CHIP, MON, PUMP |
| Short | DOGE, FARTCOIN, GRASS, SOL, BNB, ZEC, ETH, BTC |

- 16 × $12,50; ekuitas $199,87 setelah fee.
- Ledger awal: 6.528 baris bootstrap (8–24 Okt).
- F14 terendah: SAND −5,5%, SOPH −5,0%, MINA −2,5%. Ketiganya tidak masuk universe top-50 HYPE.

---

## 11. Runbook dan troubleshooting

| Gejala | Cek | Tindakan |
|---|---|---|
| Watcher mati / runs.csv basi | `state/runs.csv` (baris terakhir), Actions → FCL bot | Watchdog menyalakan watcher otomatis. Manual: `gh workflow run bot.yml --repo daijobudesu69/Crypto-FCL -f mode=loop` |
| Run "cancelled" di daftar Actions | Normal: concurrency group `fcl-state` hanya menyimpan 1 run antre; yang antre lebih lama dibatalkan | Tidak perlu tindakan selama ada run `in_progress` |
| Tidak ada baris `predicted` baru | `state/funding_pending.json` (`seen_utc` terbaru), log siklus | Cek API: `predictedFundings` masih memuat `BinPerp`? Kalau format berubah, perbaiki `hype.predicted_fundings` |
| Cakupan F14 < 70% (koin hilang dari universe) | `state/targets.csv` kolom `coverage` | Kalau bot mati > 3 hari: `gh workflow run bootstrap.yml -f days=16` (baris yang ada tidak ditimpa) |
| Rebalance tidak terjadi | `paper.json` → `last_rebalance_day`; log `[daily]` | `rebalance_due` butuh `now ≥ 08:20 UTC`. Kalau terlewat, siklus berikutnya mengejar di hari yang sama |
| "universe N < 12: flat" | Banyak candle/funding gagal | Cek error 429/HTTP di log; `hype.InfoClient` retry 6×. Siklus berikutnya mengulang |
| Error HTTP 429 | Weight > 1200/menit | Turunkan `hype.weight_per_minute` di config.yaml |
| Konflik state saat push | `save_state.sh` rebase 5× (CSV merge=union, JSON versi run ini menang) | Kalau gagal terus: cek penanda konflik di `state/` |
| UnicodeEncodeError di Windows | Konsol cp1252 vs emoji | Sudah diatasi dengan `sys.stdout.reconfigure(utf-8)` di run_*.py |
| Workflow gagal "workflow file issue" | YAML: `- run: ... "teks: ..."` (titik dua + spasi di scalar polos) | Pakai blok `run: \|` (bug ini sudah terjadi dan diperbaiki di bootstrap.yml) |
| Mode live tidak berjalan | `control.py` memetakan live/manage/flatten ke paper + peringatan | Wallet belum dipasang (lihat §13) |
| Telegram tidak terkirim | Secret kosong → pesan hanya dicetak di log | Isi secret `TELEGRAM_BOT_TOKEN` dan `TELEGRAM_CHAT_ID` di repo |
| Ingin cek cepat | `python run_status.py` (setelah `git pull`) | — |
| Ingin uji tanpa menyentuh state | `FCL_STATE_DIR=<folder sementara> python run_cycle.py` (salin `funding_ledger.csv` dulu) | — |

### Validasi berkala yang direncanakan
1. **Minggu 4:** bandingkan baris `predicted` dengan funding settled Binance dari arsip bulanan (data lake: `python scripts/binance_download.py um_funding`). Target korelasi ≥ 0,9.
2. **Bulan 2–3:** jalankan `fcl.run` riset untuk periode paper (perlu update data lake + `build_cache.py` dengan `ALPHA_FORWARD=1`), lalu bandingkan dengan `state/equity.csv`.
3. **Rem paper:** DD > 25% atau Sharpe rolling 90 hari < 0 → review.

---

## 12. Risiko dan isu yang diketahui

1. **Periode forward sangat ramah** (alt EW −57%; short alt yang crowded dibayar dua kali: lewat harga dan funding). Ekspektasi Sharpe 1–1,5.
2. **DSR IS hanya 0,3–0,6** di tingkat desk (±900 config). Forward yang kuat (DSR 0,83) adalah bukti utamanya, tapi sampelnya hanya 21 bulan.
3. **Survivorship bias:** daftar koin = yang masih hidup per 2026. Sudah diuji dengan panel 776 koin termasuk delisting (§8): Sharpe 1,33–1,60 vs 2,0–2,25 untuk koin yang masih hidup saja. Edge tetap ada, tapi angka backtest perlu dipotong ±0,6.
4. **Konsentrasi:** di IS, top-5 koin = 53–59% PnL dan 10 hari terbaik = 40%.
5. **Squeeze short:** coin yang di-short bisa naik 50–100% dalam sehari (DOGE 2021-01, GALA 2021-11). Mitigasi: filter momentum, stop +40%, dan k ≥ 8 nama.
6. **Rekaman predicted vs settled** belum divalidasi (§11). Bootstrap 14 hari pertama hanya berkorelasi ranking 0,77.
7. **Universe bot ≠ backtest** (paritas 94–96%).
8. **Stop di paper** dicek per 10 menit dengan mid, jadi bisa lebih baik atau lebih buruk daripada trigger live.
9. **Funding HYPE** punya komponen bunga ±11%/tahun untuk long; netral di buku dollar-neutral, dan short menerima.
10. **$200 dan minimum order $10:** k = 8 di $200. Kalau ekuitas < $100, k tetap 4 dan posisi < $12,50 (masih ≥ $10 sampai ekuitas $80). Di bawah itu order ditolak.

---

## 13. Menyusul (rencana)

- **Telegram:** secret `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`. `bot.yml` dan `watchdog.yml` sudah meneruskan env-nya.
- **Google Sheets:** salin `rntbot/sheets.py` dari RNT, panggil di `store.append` untuk log `orders`, `equity`, `stops`. Secret mengikuti RNT (`GOOGLE_SERVICE_ACCOUNT_JSON`, `GSHEET_SPREADSHEET_ID`).
- **Wallet / live:** ikuti pola RNT (`rntbot/live.py`, `canary.py`, `docs/ARSITEKTUR.md`):
  - API wallet di **subaccount khusus FCL**, cross margin, leverage efektif ±1×;
  - stop sebagai **order trigger** reduce-only di exchange;
  - mode `live/manage/flatten` di `control.py`;
  - canary order ±$10 sebelum live;
  - minimum order HYPE $10 dan `szDecimals` dari `meta()`.

---

## 14. Peta file (di mana menemukan apa)

| Apa | Lokasi |
|---|---|
| Bot, state, workflow | `C:\Crypto data\Crypto-FCL` / GitHub `daijobudesu69/Crypto-FCL` |
| Dokumen ini | `docs/FCL_HANDOVER.md` |
| Arsip riset (kode + hasil kecil) | `research/alpha_20261024/` di repo ini |
| Riset yang bisa dijalankan (dengan cache 313 MB) | `C:\Crypto data\backtest data and more\alpha_20261024\` |
| Data lake | `C:\Crypto data\backtest data and more\data\` (update: `scripts/update_all.sh`) |
| Laporan 1 (alpha hunt, CHOP-30, saran FCL) | `research/Alpha Hunt CHOP-30 20261024.md` (asli di `backtest data and more/`) |
| Laporan 2 (RNT, fat tail, 4 uji) | `research/Alpha Hunt - Perbandingan RNT, Fat Tail, Uji Hipotesis 20261024.md` |
| Brief riset untuk agen | `research/alpha_20261024/HUNT_BRIEF.md` |
| Hasil semua hunter | `research/alpha_20261024/hunt/hunt_results.json`, `hunt_results_r2.json`, `hunt/*/RESULTS.md` |
| Verifikasi | `research/alpha_20261024/verify/VERIFY_SUMMARY_R1.md`, `verify_r2/verify_r2_results.json` |
| Spec beku + hash | `research/alpha_20261024/SPEC_CHOP30_v1.md`, `final/frozen_hashes.txt` |
| Hasil forward | `research/alpha_20261024/final/forward_results.json`, `forward_equity.csv` |
| Statistik lanjutan | `research/alpha_20261024/stats/analysis_results.json` |
| Chart | `research/alpha_20261024/report_assets/*.png` |
| Strategi lain user | RNT `C:\Crypto data\Crypto-RS-Neutral & Trend`, RMF `...\Crypto-Momentum Rotations+Flush Basket`, MEX `...\Crypto-MEX` |

---

## 15. Glosarium

| Istilah | Arti |
|---|---|
| **F14** | Funding Binance kumulatif 14 hari (fraksi; 0,01 = 1%) |
| **Crowding** | Banyak long ber-leverage yang membayar funding tinggi |
| **IS / forward** | In-sample 2020–2024 / out-of-sample 2025-01 → 2026-10 |
| **DSR** | Deflated Sharpe Ratio: probabilitas Sharpe bukan hasil keberuntungan dari N percobaan |
| **Tercile / k** | Sepertiga universe / jumlah koin per sisi |
| **predicted / bootstrap** | Rate dari snapshot API HYPE / rekonstruksi dari arsip premium index |
| **BinPerp** | Venue Binance perp di `predictedFundings` HYPE |
