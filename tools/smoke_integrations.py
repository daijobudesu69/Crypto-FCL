"""Smoke test Telegram dan Google Sheets memakai secret repo. Manual (smoke.yml).

Telegram : kirim satu pesan uji lewat fclbot.notify.send_now.
Sheets   : login service account, baca daftar tab, buat tab sementara "_smoke",
           tulis + baca balik satu sel, lalu hapus tab itu. Isi sheet lain tidak disentuh.
Error dilaporkan dengan tipe/status saja (URL dan respons bisa memuat token).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fclbot import notify  # noqa: E402

BASE = "https://sheets.googleapis.com/v4/spreadsheets/{id}"
TAB = "_smoke"


def telegram() -> bool:
    if not notify.configured():
        print("[telegram] GAGAL: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID kosong")
        return False
    ok = notify.send_now(f"✅ <b>FCL smoke test</b>\nTelegram OK · {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%MZ}")
    print("[telegram]", "OK, pesan terkirim" if ok else "GAGAL, lihat log di atas")
    return ok


def sheets() -> bool:
    raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
    sid = os.environ.get("GSHEET_SPREADSHEET_ID", "").strip()
    if not (raw and sid):
        print("[sheets] GAGAL: GOOGLE_SERVICE_ACCOUNT_JSON / GSHEET_SPREADSHEET_ID kosong")
        return False
    try:
        info = json.loads(raw)
    except ValueError:
        print("[sheets] GAGAL: GOOGLE_SERVICE_ACCOUNT_JSON bukan JSON valid")
        return False
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    try:
        creds = service_account.Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/spreadsheets"])
        creds.refresh(Request())
    except Exception as e:  # noqa: BLE001
        print(f"[sheets] GAGAL login service account: {type(e).__name__}")
        return False
    print(f"[sheets] login OK sebagai {info.get('client_email')}")
    h = {"Authorization": f"Bearer {creds.token}"}
    url = BASE.format(id=sid)

    def call(method, path, **kw):
        r = requests.request(method, url + path, headers=h, timeout=25, **kw)
        if r.status_code >= 400:
            msg = ""
            try:
                msg = r.json().get("error", {}).get("message", "")
            except ValueError:
                pass
            raise RuntimeError(f"HTTP {r.status_code} {msg[:150]}")
        return r.json()

    gid = None
    try:
        meta = call("GET", "?fields=properties.title,sheets.properties")
        tabs = [s["properties"]["title"] for s in meta["sheets"]]
        print(f"[sheets] buka OK: {meta['properties']['title']!r}, tab: {tabs}")
        if TAB in tabs:
            gid = next(s["properties"]["sheetId"] for s in meta["sheets"] if s["properties"]["title"] == TAB)
        else:
            res = call("POST", ":batchUpdate", json={"requests": [{"addSheet": {"properties": {"title": TAB}}}]})
            gid = res["replies"][0]["addSheet"]["properties"]["sheetId"]
        stamp = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        call("PUT", f"/values/{TAB}!A1?valueInputOption=RAW", json={"values": [[stamp]]})
        back = call("GET", f"/values/{TAB}!A1")["values"][0][0]
        ok = back == stamp
        print("[sheets] tulis+baca", "OK" if ok else f"TIDAK COCOK ({back!r})")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[sheets] GAGAL: {e}")
        return False
    finally:
        if gid is not None:
            try:
                call("POST", ":batchUpdate", json={"requests": [{"deleteSheet": {"sheetId": gid}}]})
                print(f"[sheets] tab {TAB} dihapus")
            except Exception as e:  # noqa: BLE001
                print(f"[sheets] peringatan: tab {TAB} gagal dihapus: {e}")


if __name__ == "__main__":
    t, s = telegram(), sheets()
    print(f"\nHASIL  telegram={'OK' if t else 'GAGAL'}  sheets={'OK' if s else 'GAGAL'}")
    sys.exit(0 if (t and s) else 1)
