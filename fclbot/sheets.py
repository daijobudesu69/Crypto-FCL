"""Mirror log forward test ke Google Sheets. CSV di state/ tetap catatan resmi.

* store.append() menaruh baris di antrean (state/sheets_pending.json); run_cycle memanggil
  flush() sekali per siklus, jadi satu panggilan API per tab, bukan per baris.
* Gagal kirim tidak pernah menggagalkan run: antrean disimpan dan dicoba lagi siklus berikutnya.
* Tab belum ada -> dibuat dengan header dan diisi seluruh isi CSV (riwayat ikut masuk);
  baris antrean untuk tab itu dibuang karena sudah ada di CSV.
* Tanpa secret, tidak ada yang terjadi. Error dilaporkan dengan tipe/status saja.
"""
from __future__ import annotations

import json
import os

import requests

from . import store

MIRRORED = ("equity", "orders", "stops", "targets")
PENDING = "sheets_pending.json"
BASE = "https://sheets.googleapis.com/v4/spreadsheets/{id}"
MAX_PENDING = 5000


def configured() -> bool:
    return bool(os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip() and os.environ.get("GSHEET_SPREADSHEET_ID", "").strip())


def enqueue(log: str, row: list) -> None:
    if log not in MIRRORED or not configured():
        return
    q = store.load_json(PENDING, {}) or {}
    q.setdefault(log, []).append(row)
    q[log] = q[log][-MAX_PENDING:]
    store.save_json(PENDING, q)


def _token() -> str:
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"]), scopes=["https://www.googleapis.com/auth/spreadsheets"])
    creds.refresh(Request())
    return creds.token


def _call(method: str, url: str, token: str, **kw):
    r = requests.request(method, url, headers={"Authorization": f"Bearer {token}"}, timeout=25, **kw)
    if r.status_code >= 400:
        raise RuntimeError(f"HTTP {r.status_code}")
    return r.json()


def _csv_rows(log: str) -> list[list]:
    rows = store.read(log)
    return [[r[c] for c in store.COLUMNS[log]] for r in rows]


def flush() -> int:
    """Kirim antrean. Return jumlah baris yang masih tertahan."""
    if not configured():
        return 0
    q = store.load_json(PENDING, {}) or {}
    if not q:
        return 0
    try:
        sid = os.environ["GSHEET_SPREADSHEET_ID"].strip()
        url = BASE.format(id=sid)
        token = _token()
        meta = _call("GET", url + "?fields=sheets.properties.title", token)
        tabs = {s["properties"]["title"] for s in meta["sheets"]}
        left = {}
        for log, rows in q.items():
            if log not in tabs:
                _call("POST", url + ":batchUpdate", token, json={"requests": [{"addSheet": {"properties": {"title": log}}}]})
                seed = [store.COLUMNS[log]] + _csv_rows(log)
                _call("PUT", f"{url}/values/{log}!A1?valueInputOption=RAW", token, json={"values": seed})
                print(f"[sheets] tab {log} dibuat, {len(seed) - 1} baris riwayat")
                continue
            try:
                _call("POST", f"{url}/values/{log}!A1:append?valueInputOption=RAW&insertDataOption=INSERT_ROWS", token,
                      json={"values": rows})
                print(f"[sheets] {log}: {len(rows)} baris")
            except Exception as e:  # noqa: BLE001
                print(f"[sheets] {log} gagal ({e}); dicoba lagi")
                left[log] = rows
    except Exception as e:  # noqa: BLE001
        print(f"[sheets] gagal ({type(e).__name__}: {e}); {sum(map(len, q.values()))} baris tertahan")
        return sum(map(len, q.values()))
    store.save_json(PENDING, left)
    return sum(map(len, left.values()))
