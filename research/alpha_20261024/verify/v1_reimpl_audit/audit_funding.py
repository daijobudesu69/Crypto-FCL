"""Audit (a): fund panel attribution vs raw Binance funding files.

Raw files are read with a pyarrow row filter ts < 2025-01-01 so no 2025+ row is ever materialised.
Checks:
  1. timestamp jitter (offset from the nearest full hour) -> can round('1min') misplace a settlement?
  2. recompute the daily (ts, ts+1d] sum independently and compare with lab.load('fund','1d')
  3. the 00:00 settlement at the trade time: which bar holds it, and is it charged to W[t] or W[t-1]?
  4. F14 at row t contains only settlements with time <= close of bar t
  5. funding-interval changes (8h -> 4h / 1h) in 2020-24 for the spot-check coins
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab

RAW = r"C:\Crypto data\backtest data and more\data\binance\um\funding"
CUT = pd.Timestamp("2025-01-01", tz="UTC")
F = lab.load("fund", "1d")
C = lab.load("close", "1d")
SYMS = ["BTCUSDT", "ZECUSDT", "LITUSDT", "TRXUSDT", "SOLUSDT", "1000PEPEUSDT"]


def raw(sym):
    df = pd.read_parquet(os.path.join(RAW, f"{sym}.parquet"), filters=[("ts", "<", CUT)])
    assert df["ts"].max() < CUT
    return df


all_jit = []
for s in SYMS:
    if s not in F.columns:
        print(s, "not in panel"); continue
    d = raw(s).dropna(subset=["funding_rate"])
    t = d["ts"].astype("datetime64[ns, UTC]")
    off = (t - t.dt.round("1h")).dt.total_seconds()
    all_jit.append(off.abs().max())
    # own attribution: settlement exactly at D+1 00:00 belongs to bar D; settlement at D 00:00 belongs to bar D-1
    tr = t.dt.round("1h")                       # exact settlement hour
    at_midnight = ((tr.dt.hour == 0) & (tr.dt.minute == 0)).astype("int64")
    bar = tr.dt.floor("1D") - pd.to_timedelta(at_midnight, unit="D")
    # (equivalent to: bar = ceil(t) - 1D for settlements on the hour)
    own = d.groupby(bar.values)["funding_rate"].sum()
    own.index = pd.DatetimeIndex(own.index).tz_convert("UTC") if pd.DatetimeIndex(own.index).tz else pd.DatetimeIndex(own.index).tz_localize("UTC")
    p = F[s].astype("float64")
    j = pd.concat([own.rename("own"), p.rename("panel")], axis=1)
    j = j[(j.index >= "2021-01-01") & (j.index < "2024-12-31")]
    both = j.dropna()
    mism = (both["own"] - both["panel"]).abs() > 1e-6
    only_panel = j["own"].isna() & j["panel"].notna() & (j["panel"] != 0)
    only_raw = j["own"].notna() & j["panel"].isna()
    iv = d.loc[(t >= "2020-01-01"), "funding_interval_hours"].value_counts().to_dict()
    print(f"{s:12s} n_settle={len(d):5d} max|offset from hour|={off.abs().max():.3f}s  "
          f"days compared={len(both)} mismatches={int(mism.sum())} panel-only={int(only_panel.sum())} raw-only={int(only_raw.sum())} "
          f"interval_hours={iv}")
    if mism.any():
        print(both[mism].head())
    # show one concrete day with its raw settlements
    if s in ("BTCUSDT", "LITUSDT"):
        for day in ["2021-05-19", "2023-03-10"]:
            D = pd.Timestamp(day, tz="UTC")
            win = d[(t > D - pd.Timedelta("1h")) & (t <= D + pd.Timedelta("1D") + pd.Timedelta("1h"))]
            print(f"   {s} raw settlements around bar {day}:")
            for _, r in win.iterrows():
                print(f"      {r['ts']}  rate={r['funding_rate']:+.6f}")
            print(f"   panel fund[{day}] = {p.loc[D]:+.6f}   panel fund[{(D - pd.Timedelta('1D')).date()}] = {p.loc[D - pd.Timedelta('1D')]:+.6f}")

print("max jitter over spot-check coins (s):", max(all_jit))

# timing statement check: lab.backtest charges W[t] with F[t+1]. F[t+1] = settlements in (t+1 00:00, t+2 00:00].
# W[t] is traded at close t = (t+1) 00:00. The 00:00 settlement AT the trade time is in F[t] -> charged to W[t-1]
# (the position that was held into 00:00). Verify with a synthetic panel through lab.backtest's own code path.
ix = C.index
W = pd.DataFrame(0.0, index=ix, columns=C.columns)
t0 = pd.Timestamp("2023-03-10", tz="UTC")
W.loc[t0, "BTCUSDT"] = -1.0                       # short 1x BTC for exactly one bar (decided at close of 03-10)
res = lab.backtest(W, "1d", cost=0.0)
row = res["df"].loc[t0 + pd.Timedelta("1D")]
print("synthetic 1-bar short BTC decided at close 2023-03-10 -> lab fund_r =", f"{row['fund_r']:+.6f}",
      " expected +F[2023-03-11] =", f"{F.loc[t0 + pd.Timedelta('1D'), 'BTCUSDT']:+.6f}",
      " (NOT F[2023-03-10] =", f"{F.loc[t0, 'BTCUSDT']:+.6f})")
print("price leg:", f"{row['gross_r']:+.6f}", " expected -(C[03-11]/C[03-10]-1) =",
      f"{-(C.loc[t0 + pd.Timedelta('1D'), 'BTCUSDT'] / C.loc[t0, 'BTCUSDT'] - 1):+.6f}")
# F14 at row t: rolling(14) over F rows t-13..t => settlements in (t-13 00:00, t+1 00:00] <= close of bar t. ok by construction.

# 2. coverage: are there listed days with zero funding although the coin has a raw settlement?  (whole panel)
z = (F == 0) & C.notna()
print("listed coin-days with fund == 0 (no settlement inside bar):", int(z.values.sum()), " of", int(C.notna().values.sum()))
zz = z.sum().sort_values(ascending=False).head(8)
print(zz.to_dict())
