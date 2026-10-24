"""Ringkasan cepat tanpa mengubah state: ekuitas paper, posisi, dan ranking F14 saat ini.

    python run_status.py
"""
from __future__ import annotations

import datetime as dt
import sys

from fclbot import ledger, store


def main() -> None:
    now = dt.datetime.now(dt.timezone.utc)
    st = store.load_json("paper.json") or {}
    print(f"ekuitas paper: {st.get('equity', '-')}  (mulai {st.get('started_utc', '-')}, rebalance terakhir {st.get('last_rebalance_day', '-')})")
    for coin, p in sorted((st.get("positions") or {}).items(), key=lambda kv: kv[1]["qty"]):
        print(f"  {coin:10s} {'LONG ' if p['qty'] > 0 else 'SHORT'} {abs(p['qty'] * p['last_px']):8.2f} USD  entry {p['entry_px']}")
    f = ledger.f14(now)
    rows = sorted(((c, s, cov) for c, (s, cov) in f.items() if s is not None), key=lambda x: x[1])
    print(f"\nF14 (funding Binance 14 hari) untuk {len(rows)} koin, cakupan dalam kurung:")
    for c, s, cov in rows[:8]:
        print(f"  rendah  {c:10s} {s * 100:+.3f}%  ({cov:.0%})")
    for c, s, cov in rows[-8:]:
        print(f"  tinggi  {c:10s} {s * 100:+.3f}%  ({cov:.0%})")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # konsol Windows (cp1252) + emoji
    main()
