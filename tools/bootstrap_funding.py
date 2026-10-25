"""Isi ledger funding 14+ hari ke belakang supaya paper test bisa mulai tanpa menunggu 14 hari.

Rekonstruksi dari arsip premium index Binance; rumus dan validasinya di fclbot/bootstrap.py.
Rekaman bot (source=predicted) menggantikan baris ini begitu tersedia, dan setelah 14 hari
bootstrap tidak dipakai lagi. Bot juga memanggil fclbot.bootstrap otomatis kalau
predictedFundings HYPE berhenti (lihat jobs.py).

    python tools/bootstrap_funding.py            # 16 hari terakhir yang ada di arsip
    python tools/bootstrap_funding.py --days 20
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fclbot import bootstrap, config, hype  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=16)
    a = ap.parse_args()
    cfg = config.load()
    info = hype.InfoClient(cfg.hype["info_url"], cfg.hype["weight_per_minute"])
    try:
        iv = {c: v["interval_h"] for c, v in info.predicted_fundings().items() if c in cfg.coins}
    except Exception as e:  # noqa: BLE001
        print(f"[bootstrap] predictedFundings gagal ({type(e).__name__}); interval dari ledger")
        iv = {}
    if not iv:
        iv = {c: bootstrap.known_intervals().get(c, 8) for c in cfg.coins}
    now = dt.datetime.now(dt.timezone.utc)
    days = [now.date() - dt.timedelta(days=d) for d in range(a.days, 0, -1)]
    n_rows, _ = bootstrap.fill(cfg, days, iv, now)
    print(f"[bootstrap] selesai: {n_rows} baris")
    return 0


if __name__ == "__main__":
    sys.exit(main())
