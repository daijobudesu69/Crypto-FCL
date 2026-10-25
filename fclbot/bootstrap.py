"""Rekonstruksi funding Binance dari arsip premium index harian (data.binance.vision).

Dipakai dua jalur:
  * tools/bootstrap_funding.py (manual / bootstrap.yml): isi 16 hari sebelum bot mulai.
  * jobs.run (otomatis): cadangan kalau predictedFundings HYPE berhenti memberi BinPerp.

Binance tidak menyediakan arsip funding harian (hanya bulanan), tetapi arsip premium index
1 jam harian ada. Funding Binance dihitung dari premium index:
    F = P + clamp(I - P, -0,05%, +0,05%),   P = rata-rata premium selama interval,
    I = 0,01% per 8 jam (diskalakan ke interval funding koin).
Validasi di data lake (research/alpha_20261024/stats/archive_signal_study.py): korelasi ranking
F14 harian antar koin vs funding settled asli 0,90 (2020-24) / 0,81 (2025-26); koin short top-8
yang sama 75% / 64%. Sinyal 100% arsip: Sharpe forward 2,33 -> 1,08-1,19, IS 1,64 -> 1,53-1,81.
Layak sebagai cadangan sementara, bukan pengganti. Baris ditulis sebagai source=bootstrap; rekaman bot (predicted)
selalu menang kalau keduanya ada. Arsip satu hari baru terbit setelah hari itu selesai.
"""
from __future__ import annotations

import datetime as dt
import io
import zipfile

import requests

from . import ledger, store


def premium(base: str, sym: str, day: dt.date, session=None) -> dict:
    """{open_time_ms: close premium jam itu}; {} kalau arsip hari itu belum/tidak ada."""
    url = f"{base}/{sym}/1h/{sym}-1h-{day.isoformat()}.zip"
    r = (session or requests).get(url, timeout=30)
    if r.status_code != 200:
        return {}
    z = zipfile.ZipFile(io.BytesIO(r.content))
    out = {}
    for line in z.read(z.namelist()[0]).decode().splitlines():
        parts = line.split(",")
        if not parts[0].isdigit():
            continue
        out[int(parts[0])] = float(parts[4])
    return out


def known_intervals() -> dict:
    """{coin: interval_h} terakhir di ledger, untuk saat predictedFundings tidak tersedia."""
    out = {}
    for r in store.read("funding_ledger"):
        out[r["coin"]] = int(float(r["interval_h"]))
    return out


def fill(cfg, days: list, intervals: dict, now: dt.datetime, log=print) -> tuple[int, list]:
    """Tulis settlement hasil rekonstruksi untuk `days` (list date). Baris yang sudah ada dilewati.

    Return (jumlah baris, hari yang arsipnya tersedia untuk >= separuh koin)."""
    base = cfg.bootstrap["archive_url"]
    have = {(r["coin"], int(r["settle_ms"])) for r in store.read("funding_ledger")}
    now_iso = now.isoformat(timespec="seconds")
    s = requests.Session()
    n_rows = 0
    got = {d: 0 for d in days}
    coins = [(c, sym) for c, sym in sorted(cfg.coins.items()) if c in intervals]
    for coin, sym in coins:
        ih = intervals[coin]
        prem = {}
        for d in days:
            p = premium(base, sym, d, s)
            if p:
                got[d] += 1
                prem.update(p)
        if not prem:
            log(f"[bootstrap] {coin}: arsip kosong")
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
        log(f"[bootstrap] {coin} ({sym}, {ih}h): {rows} settlement")
    complete = [d for d in days if coins and got[d] >= len(coins) / 2]
    return n_rows, complete
