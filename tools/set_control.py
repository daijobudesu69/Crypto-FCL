"""Tulis ulang control/bot.yaml. Dipanggil tools/save_control.sh (workflow control.yml).

    python tools/set_control.py --mode paper
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fclbot import control  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="tetap")
    a = ap.parse_args()
    cur = control.read((control.PATH,))
    m = cur.mode if a.mode == "tetap" else a.mode
    if m not in control.MODES:
        print(f"[control] mode tidak dikenal: {m}")
        return 2
    os.makedirs(os.path.dirname(control.PATH), exist_ok=True)
    with open(control.PATH, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(control.render(m))
    print(f"[control] mode={m}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
