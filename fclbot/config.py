"""Baca config.yaml dan coins.csv (daftar 91 koin beku: nama Binance -> nama HYPE)."""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class Config:
    strategy: dict
    paper: dict
    schedule: dict
    hype: dict
    bootstrap: dict
    coins: dict = field(default_factory=dict)      # hype name -> binance symbol

    @property
    def rebalance_hm(self) -> tuple[int, int]:
        h, m = str(self.schedule["rebalance_utc"]).split(":")
        return int(h), int(m)


def load(path: str | None = None, coins_path: str | None = None) -> Config:
    with open(path or os.path.join(ROOT, "config.yaml"), encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    coins = {}
    with open(coins_path or os.path.join(ROOT, "coins.csv"), encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            coins[row["hype"]] = row["binance"]
    return Config(strategy=doc["strategy"], paper=doc["paper"], schedule=doc["schedule"], hype=doc["hype"],
                  bootstrap=doc.get("bootstrap", {}), coins=coins)
