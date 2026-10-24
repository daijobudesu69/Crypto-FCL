"""Aturan FCL v1.0 (sama dengan alpha_20261024/fcl.py yang di-backtest dan di-forward-test).

Universe : koin di coins.csv yang sudah trading di HYPE >= 60 hari (volume > 0), top 50 menurut
           median volume USD 30 hari (candle harian HYPE), dengan cakupan funding >= 70%.
Sinyal   : F14 = funding Binance 14 hari (ledger).
Short    : k koin F14 tertinggi, kecuali koin yang return 14 harinya di 10% teratas universe
           dan koin yang kena stop di hari sebelumnya.
Long     : k koin F14 terendah (bukan koin yang short).
k        : min(floor(N/3), max(4, floor(ekuitas * gross / 2 / 12,5))); N < 12 -> flat.
Bobot    : sama rata, +gross/2 sisi long, -gross/2 sisi short.

Beda dengan backtest (dicatat di README): ranking likuiditas memakai volume HYPE (backtest:
volume Binance), F14 dari snapshot predicted (backtest: settled), rebalance 08:20 UTC.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Book:
    weights: dict          # coin -> bobot (fraksi ekuitas)
    table: list            # baris untuk state/targets.csv
    k: int
    n_universe: int
    note: str = ""


def universe(candles: dict, f14: dict, p: dict) -> pd.DataFrame:
    rows = []
    for coin, df in candles.items():
        if df is None or df.empty:
            continue
        traded = df[df["volume"] > 0]
        if len(traded) < p["min_history_days"]:
            continue
        last = df.iloc[-1]
        if not last["volume"] > 0:
            continue
        usd = (df["volume"] * df["close"]).tail(p["volume_days"])
        usd = usd[usd > 0]
        if len(usd) < 15:
            continue
        closes = df["close"].values
        mom = closes[-1] / closes[-1 - p["mom_days"]] - 1 if len(closes) > p["mom_days"] else np.nan
        sig, cov = f14.get(coin, (None, 0.0))
        rows.append({"coin": coin, "dqv": float(usd.median()), "mom14": mom,
                     "f14": sig, "coverage": cov})
    u = pd.DataFrame(rows, columns=["coin", "dqv", "mom14", "f14", "coverage"])
    if u.empty:
        return u
    u = u.sort_values(["dqv", "coin"], ascending=[False, True]).head(p["universe_top"])
    u = u[(u["coverage"] >= p["min_coverage"]) & u["f14"].notna()].copy()
    return u.reset_index(drop=True)


def targets(candles: dict, f14: dict, equity: float, banned: set, p: dict) -> Book:
    u = universe(candles, f14, p)
    N = len(u)
    if N < 3 * p["k_min"]:
        return Book({}, [], 0, N, note=f"universe {N} < {3 * p['k_min']}: flat")
    k = int(min(N // 3, max(p["k_min"], math.floor(equity * p["gross"] / 2 / p["slot_usd"]))))
    u["mom_rank"] = u["mom14"].rank(pct=True)
    cand_s = u[(u["mom_rank"] <= p["mom_skip_pct"]) | u["mom_rank"].isna()]
    cand_s = cand_s[~cand_s["coin"].isin(banned)]
    short = cand_s.sort_values(["f14", "coin"], ascending=[False, True]).head(k)["coin"].tolist()
    long_ = u[~u["coin"].isin(short)].sort_values(["f14", "coin"]).head(k)["coin"].tolist()
    w = {}
    if short and long_:
        for c in short:
            w[c] = -p["gross"] / 2 / len(short)
        for c in long_:
            w[c] = p["gross"] / 2 / len(long_)
    table = []
    for r in u.itertuples():
        side = "short" if r.coin in short else ("long" if r.coin in long_ else "")
        table.append({"coin": r.coin, "f14_pct": r.f14 * 100, "coverage": r.coverage, "mom14": r.mom14,
                      "mom_rank": r.mom_rank, "dqv_musd": r.dqv / 1e6, "side": side, "w": w.get(r.coin, 0.0)})
    return Book(w, table, k, N)
