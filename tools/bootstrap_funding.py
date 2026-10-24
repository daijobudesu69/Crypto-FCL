"""Isi ledger funding 14+ hari ke belakang supaya paper test bisa mulai tanpa menunggu 14 hari.

Binance tidak menyediakan arsip funding harian (hanya bulanan), tetapi arsip premium index
1 jam harian ada di data.binance.vision. Funding Binance dihitung dari premium index:
    F = P + clamp(I - P, -0,05%, +0,05%),   P = rata-rata premium selama interval,
    I = 0,01% per 8 jam (diskalakan ke interval 4 jam = 0,005%).
Validasi di data lake (September 2026, 92 koin): korelasi Pearson jumlah bulanan 0,94,
korelasi ranking 0,77. Cukup untuk mengisi celah; rekaman bot (source=predicted) menggantikan
baris ini begitu tersedia, dan setelah 14 hari bootstrap tidak dipakai lagi.

    python tools/bootstrap_funding.py            # 16 hari terakhir yang ada di arsip
    python tools/bootstrap_funding.py --days 20
"""
from __future__ import annotations

import argparse
import datetime as dt
import io
import os
import sys
import zipfile

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fclbot import config, hype, ledger, store  # noqa: E402


def premium(base: str, sym: str, day: dt.date) -> dict:
    url = f"{base}/{sym}/1h/{sym}-1h-{day.isoformat()}.zip"
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return {}
    z = zipfile.ZipFile(io.BytesIO(r.content))
    out = {}
    for line in z.read(z.namelist()[0]).decode().splitlines():
        parts = line.split(",")
        if not parts[0].isdigit():
            continue
        out[int(parts[0])] = float(parts[4])        # open_time -> close premium of that hour
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=16)
    a = ap.parse_args()
    cfg = config.load()
    info = hype.InfoClient(cfg.hype["info_url"], cfg.hype["weight_per_minute"])
    iv = {c: v["interval_h"] for c, v in info.predicted_fundings().items() if c in cfg.coins}
    today = dt.datetime.now(dt.timezone.utc).date()
    have = {(r["coin"], int(r["settle_ms"])) for r in store.read("funding_ledger")}
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    n_rows = 0
    for coin, sym in sorted(cfg.coins.items()):
        if coin not in iv:
            continue
        ih = iv[coin]
        prem = {}
        for d in range(a.days, 0, -1):
            prem.update(premium(cfg.bootstrap["archive_url"], sym, today - dt.timedelta(days=d)))
        if not prem:
            print(f"[bootstrap] {coin}: arsip kosong")
            continue
        hours = sorted(prem)
        step = ih * 3_600_000
        t = (hours[0] // step + 1) * step
        rows = 0
        while t <= hours[-1] + 3_600_000:
            w = [prem[h] for h in hours if t - step <= h < t]
            if len(w) >= max(1, ih // 2) and (coin, t) not in have:
                P = sum(w) / len(w)
                interest = 0.0001 * ih / 8
                rate = P + min(max(interest - P, -0.0005), 0.0005)
                store.append("funding_ledger", {"settle_utc": ledger.iso(t), "settle_ms": t, "coin": coin, "rate": rate,
                                                "interval_h": ih, "source": "bootstrap", "seen_utc": now_iso})
                rows += 1
            t += step
        n_rows += rows
        print(f"[bootstrap] {coin} ({sym}, {ih}h): {rows} settlement")
    print(f"[bootstrap] selesai: {n_rows} baris")
    return 0


if __name__ == "__main__":
    sys.exit(main())
