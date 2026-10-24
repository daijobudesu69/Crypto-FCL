"""State JSON dan log CSV di folder state/ (di-commit ke repo oleh workflow).

* JSON ditulis atomik (file sementara lalu rename): job yang mati di tengah
  penulisan tidak boleh meninggalkan file setengah jadi.
* CSV append-only. .gitattributes memakai merge=union untuk state/*.csv, jadi
  dua run yang menambah baris bersamaan tidak bentrok saat rebase.
* Mirror ke Google Sheets menyusul (lihat README); CSV di state/ adalah catatan resmi.
"""
from __future__ import annotations

import csv
import json
import os
import tempfile

from .config import ROOT

STATE_DIR = os.environ.get("FCL_STATE_DIR") or os.path.join(ROOT, "state")

# Kolom tetap per log. Menambah kolom: tambahkan di AKHIR (file lama dirotasi otomatis).
COLUMNS = {
    "runs": ["time_utc", "commit", "mode", "snapshot", "daily", "errors", "duration_s"],
    "funding_ledger": ["settle_utc", "settle_ms", "coin", "rate", "interval_h", "source", "seen_utc"],
    "orders": ["time_utc", "day", "coin", "side", "kind", "qty", "px", "notional", "fee", "reason"],
    "equity": ["time_utc", "event", "equity", "peak", "dd_pct", "gross", "net", "n_long", "n_short",
               "fees_cum", "funding_cum", "price_pnl_cum"],
    "targets": ["day", "coin", "f14_pct", "coverage", "mom14", "mom_rank", "dqv_musd", "side", "w"],
    "stops": ["time_utc", "coin", "entry_px", "trigger_px", "fill_px", "pnl"],
}


def path(name: str) -> str:
    return os.path.join(STATE_DIR, name)


def load_json(name: str, default=None):
    p = path(name)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def save_json(name: str, data) -> None:
    os.makedirs(STATE_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix=".tmp_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(data, fh, indent=1, ensure_ascii=False, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path(name))


def _rotate_if_needed(p: str, cols: list) -> None:
    if not os.path.exists(p):
        return
    with open(p, encoding="utf-8", newline="") as fh:
        head = next(csv.reader(fh), None)
    if head and head != cols:
        base, ext = os.path.splitext(p)
        i = 1
        while os.path.exists(f"{base}.v{i}{ext}"):
            i += 1
        os.replace(p, f"{base}.v{i}{ext}")


def append(log: str, row: dict, mirror: bool = True) -> None:
    cols = COLUMNS[log]
    os.makedirs(STATE_DIR, exist_ok=True)
    p = path(f"{log}.csv")
    _rotate_if_needed(p, cols)
    new = not os.path.exists(p)
    clean = {k: _fmt(row.get(k)) for k in cols}
    with open(p, "a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow(clean)


def read(log: str) -> list[dict]:
    p = path(f"{log}.csv")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return round(v, 8)
    return v


def rewrite(log: str, rows: list[dict]) -> None:
    """Tulis ulang seluruh log (dipakai untuk memangkas ledger funding lama)."""
    cols = COLUMNS[log]
    os.makedirs(STATE_DIR, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix=".tmp_", suffix=".csv")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: _fmt(r.get(k)) if not isinstance(r.get(k), str) else r.get(k) for k in cols})
    os.replace(tmp, path(f"{log}.csv"))
