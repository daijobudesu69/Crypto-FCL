"""Listing-age panels (pre-registered 2026-10-24, before any return was computed).

raw age   = days since the coin's first Binance perp daily candle in the lab cache (live: first HYPE
            candle / a static listing-date table). Coins whose first candle is before 2020-03-01 are the
            2019/early-2020 legacy listing batch (BTC ETH BCH XRP LTC TRX ETC LINK XLM ADA XMR DASH ZEC
            ATOM BNB IOST): their first candle is not a token launch, so they are treated as MATURE.
clean age = raw age, but coins whose token traded publicly >= ~10 months before their Binance perp
            listing (old tokens listed late, renames, migrations) are also treated as MATURE.
            The list is static public metadata (token launch years), not a market-data signal:
            ALGO DOGE EGLD CVC ZEN AAVE(LEND) COTI HBAR ONE CELR AR GALA INJ LDO QNT FET T MINA STX NMR
            PENDLE ARK CAKE STEEM ONG MOVR LSK TAO RENDER(RNDR) RARE G(GAL) SYN POL(MATIC) AERO RAYSOL
            PHA BONK VIRTUAL.
Mature coins get age = 9999.
"""
import numpy as np, pandas as pd
import r2lib
lab = r2lib.lab

LEGACY_CUTOFF = pd.Timestamp("2020-03-01", tz="UTC")
OLD_TOKENS = ["ALGO", "DOGE", "EGLD", "CVC", "ZEN", "AAVE", "COTI", "HBAR", "ONE", "CELR", "AR", "GALA",
              "INJ", "LDO", "QNT", "FET", "T", "MINA", "STX", "NMR", "PENDLE", "ARK", "CAKE", "STEEM",
              "ONG", "MOVR", "LSK", "TAO", "RENDER", "RARE", "G", "SYN", "POL", "AERO", "RAYSOL", "PHA",
              "1000BONK", "VIRTUAL"]
OLD_SYMS = [s + "USDT" for s in OLD_TOKENS]


def first_dates():
    C = lab.load("close", "1d")
    return C.apply(lambda s: s.first_valid_index())


def age_panel(kind="raw"):
    C = lab.load("close", "1d")
    first = first_dates()
    days = (C.index.values[:, None] - first.values[None, :]) / np.timedelta64(1, "D")
    A = pd.DataFrame(days, index=C.index, columns=C.columns).astype(float)
    A = A.where(C.notna())
    mature = (first < LEGACY_CUTOFF)
    if kind == "clean":
        mature = mature | pd.Series(C.columns.isin(OLD_SYMS), index=C.columns)
    A.loc[:, mature.values] = A.loc[:, mature.values].where(A.loc[:, mature.values].isna(), 9999.0)
    return A
