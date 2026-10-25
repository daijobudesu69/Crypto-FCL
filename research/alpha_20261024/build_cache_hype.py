"""HYPE (Hyperliquid) daily panels with Binance symbol names as columns, for venue-true replays.

Output: cache/hype_1d/{open,high,low,close,qv,fund}.parquet
fund = sum of hourly HYPE funding settlements in (ts, ts+1d], same convention as build_cache.py.
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
OUT = os.path.join(HERE, "cache", "hype_1d")
END = pd.Timestamp("2026-10-03", tz="UTC")


def main():
    m = pd.read_csv(os.path.join(HERE, "hype_map.csv")).dropna()
    ix = pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), END, freq="1D", inclusive="left").astype("datetime64[ns, UTC]")
    cols = pd.read_parquet(os.path.join(HERE, "cache", "1d", "close.parquet")).columns
    out = {f: {} for f in ["open", "high", "low", "close", "qv", "fund"]}
    for b, h in zip(m["binance"], m["hype"]):
        k = pd.read_parquet(os.path.join(DATA, "hyperliquid", "candles", "1d", f"{h}.parquet"))
        k["ts"] = k["ts"].astype("datetime64[ns, UTC]")
        k = k.set_index("ts").reindex(ix)
        for f, c in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"), ("qv", "quote_volume")]:
            out[f][b] = k[c].astype("float32")
        fp = os.path.join(DATA, "hyperliquid", "funding", f"{h}.parquet")
        fs = pd.Series(np.nan, index=ix, dtype="float32")
        if os.path.exists(fp):
            f = pd.read_parquet(fp)[["ts", "funding_rate"]].dropna()
            t = f["ts"].astype("datetime64[ns, UTC]").dt.round("1min")
            bar = (t - pd.Timedelta("1ms")).dt.floor("1D")
            g = f.groupby(bar.values)["funding_rate"].sum()
            gi = pd.DatetimeIndex(g.index)
            g.index = gi.tz_localize("UTC") if gi.tz is None else gi.tz_convert("UTC")
            fs = g.reindex(ix).astype("float32")
            first = g.index.min()
            fs = fs.where(fs.notna() | out["close"][b].isna() | (ix < first), 0.0)
        out["fund"][b] = fs
    os.makedirs(OUT, exist_ok=True)
    for f, d in out.items():
        df = pd.DataFrame(d, index=ix).reindex(columns=cols).astype("float32")
        df.index.name = "ts"
        df.to_parquet(os.path.join(OUT, f"{f}.parquet"), compression="zstd")
    print("done", len(m))


if __name__ == "__main__":
    main()
