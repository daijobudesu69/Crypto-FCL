"""Satu siklus bot FCL. Dipanggil watcher GitHub Actions tiap ~10 menit, atau manual:
    python run_cycle.py

Exit code 0 = sukses (termasuk "tidak ada yang perlu dikerjakan"), 1 = ada error.
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time
import traceback

from fclbot import config, control, hype, jobs, notify, sheets, store

QUIET_IDLE_MIN = 60     # baris "idle" di runs.csv paling sering 1x per jam


def main(now: dt.datetime | None = None, info=None) -> int:
    t0 = time.time()
    now = now or dt.datetime.now(dt.timezone.utc)
    cfg = config.load()
    ctrl = control.read()
    outbox = notify.Outbox(now)
    errors = []

    seen = store.load_json("alerts.json", {}) or {}
    if ctrl.problem and seen.get("control_problem") != ctrl.problem:
        outbox.add(f"⚠️ <b>FCL — control/bot.yaml</b>\n{notify.esc(ctrl.problem)}")
    if seen.get("mode") and seen.get("mode") != ctrl.mode:
        outbox.add(f"🔧 <b>FCL — mode sekarang</b>: <b>{ctrl.mode}</b>")
    if seen.get("control_problem") != ctrl.problem or seen.get("mode") != ctrl.mode:
        seen.update(control_problem=ctrl.problem, mode=ctrl.mode)
        store.save_json("alerts.json", seen)

    ctx = jobs.Ctx(cfg=cfg, info=info or hype.InfoClient(cfg.hype["info_url"], cfg.hype["weight_per_minute"]),
                   ctrl=ctrl, outbox=outbox, now=now)
    res = {}
    try:
        res = jobs.run(ctx)
        errors += res.get("errors", [])       # gangguan yang tidak menghentikan siklus (feed funding)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        msg = f"{type(e).__name__}: {str(e)[:200]}"
        errors.append(msg)
        last = seen.get("err_cycle")
        if not last or (now - dt.datetime.fromisoformat(last)).total_seconds() > 6 * 3600:
            outbox.add(f"⚠️ <b>FCL — siklus gagal</b>\n<code>{notify.esc(msg)}</code>\n<i>Dicoba ulang di siklus berikutnya.</i>")
            seen["err_cycle"] = now.isoformat()
            store.save_json("alerts.json", seen)

    pending = outbox.flush()
    sheets.flush()
    _log_run(now, ctrl, res, errors, time.time() - t0)
    if pending:
        print(f"[cycle] {pending} pesan Telegram masih tertahan")
    print(f"[cycle] selesai: snapshot={res.get('snapshot')} daily={res.get('daily')} errors={len(errors)}")
    return 1 if errors else 0


def _log_run(now, ctrl, res, errors, dur):
    busy = res.get("daily") or errors
    if not busy and os.environ.get("FCL_QUIET_IDLE") == "1":
        rows = store.read("runs")
        if rows:
            last = dt.datetime.fromisoformat(rows[-1]["time_utc"])
            if (now - last).total_seconds() < QUIET_IDLE_MIN * 60:
                return
    snap = res.get("snapshot") or {}
    store.append("runs", {"time_utc": now.isoformat(timespec="seconds"),
                          "commit": (os.environ.get("GITHUB_SHA") or "")[:10], "mode": ctrl.mode,
                          "snapshot": f"settled={snap.get('settled', '')} pending={snap.get('pending', '')}",
                          "daily": bool(res.get("daily")), "errors": " | ".join(errors)[:500],
                          "duration_s": round(dur, 1)}, mirror=False)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # konsol Windows (cp1252) + emoji
    sys.exit(main())
