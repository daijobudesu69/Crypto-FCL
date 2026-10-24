"""Kendali mode bot, dibaca SETIAP siklus dari control/bot.yaml (pola RNT).

mode:
  off    tidak melakukan apa-apa (perekaman funding juga berhenti)
  paper  rekam funding + buku paper (DEFAULT)
Mode live/manage/flatten disiapkan setelah wallet HYPE siap. Sampai itu, nilai tersebut
diperlakukan sebagai paper dan dilaporkan sebagai masalah.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import yaml

from .config import ROOT

MODES = ("off", "paper")
RESERVED = ("live", "manage", "flatten")
PATH = os.path.join(ROOT, "control", "bot.yaml")
CACHE = os.path.join(ROOT, ".fcl_control_origin.yaml")   # ditulis refresh_state.sh, di-gitignore


@dataclass
class Control:
    mode: str = "paper"
    problem: str | None = None


def _norm(v) -> str:
    if v is False:            # YAML 1.1 membaca `off` tanpa kutip sebagai boolean False
        return "off"
    return str(v or "").strip().lower()


def read(paths=None) -> Control:
    paths = paths or (CACHE, PATH)
    p = next((x for x in paths if os.path.exists(x)), None)
    if p is None:
        return Control(problem="control/bot.yaml tidak ada; dipakai paper")
    try:
        with open(p, encoding="utf-8") as fh:
            doc = yaml.safe_load(fh) or {}
        if not isinstance(doc, dict):
            raise ValueError("bukan pasangan kunci: nilai")
    except Exception as e:  # noqa: BLE001
        return Control(problem=f"control/bot.yaml tidak bisa dibaca ({type(e).__name__}); dipakai paper")
    c = Control(mode=_norm(doc.get("mode", "paper")))
    if c.mode in RESERVED:
        c.problem = f"mode '{c.mode}' belum tersedia (wallet HYPE belum disiapkan); dipakai paper"
        c.mode = "paper"
    elif c.mode not in MODES:
        c.problem = f"mode '{c.mode}' tidak dikenal; dipakai paper"
        c.mode = "paper"
    return c


def render(mode: str) -> str:
    if mode not in MODES:
        raise ValueError("mode tidak dikenal")
    return f"""\
# Kendali bot FCL. Dibaca watcher SETIAP siklus (~10 menit).
# Ubah lewat workflow (PowerShell / CMD / Git Bash):
#   gh workflow run control.yml -f mode=paper
#   gh workflow run control.yml -f mode=off
#
# mode: off | paper   (live/manage/flatten menyusul setelah wallet HYPE siap)
mode: "{mode}"
"""
