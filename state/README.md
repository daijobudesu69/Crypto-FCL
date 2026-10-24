# state/ — ditulis bot, di-commit otomatis

| File | Isi |
|---|---|
| `funding_ledger.csv` | Funding Binance per settlement (`predicted` = rekaman bot, `bootstrap` = rekonstruksi arsip). Dipangkas ke 40 hari |
| `funding_pending.json` | Snapshot terakhir untuk settlement berikutnya per koin |
| `paper.json` | Buku paper: ekuitas, posisi (qty, entry), fee/funding kumulatif, larangan short setelah stop |
| `equity.csv` | Ekuitas paper per jam (`mark`) dan saat rebalance |
| `orders.csv` | Semua order paper (open/adjust/close/stop) |
| `targets.csv` | Universe harian: F14, cakupan, momentum, sisi, bobot |
| `stops.csv` | Kejadian stop short +40% |
| `runs.csv` | Log siklus (maks 1 baris/jam saat idle) |
| `outbox.json`, `alerts.json`, `watchdog.json` | Telegram tertunda, status alarm, status watchdog |
