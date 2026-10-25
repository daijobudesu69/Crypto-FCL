"""Build wide float32 panels (index = bar open ts UTC, columns = symbols) for the alpha hunt.

Run once:  python build_cache.py
Output:    cache/{tf}/{field}.parquet  for tf in 1d, 4h, 1h

Alignment rules (all values at row t are known at the CLOSE of bar t = ts + tf):
- open/high/low/close/qv/tbqv/trades: the bar itself (partial bars with < 80% of 1m bars -> NaN)
- fund:  sum of funding settlements with time in (ts, ts+tf]  -> charged to a position held over bar t
- oi_usd, top_pos_ls, top_acct_ls, global_ls, taker_ls_vol: last 5m metric at or before close (stale > 2h -> NaN)
- prem: premium index close of the last hour inside the bar; prem_mean: mean of hourly closes inside the bar
- spot_close, spot_qv (1d only): Binance spot daily bar
"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
CACHE = os.path.join(HERE, "cache")
TFS = {"1d": "1D", "4h": "4h", "1h": "1h"}
FULL_MIN = {"1d": 1440, "4h": 240, "1h": 60}
END = pd.Timestamp("2026-10-03", tz="UTC")


def syms():
    u = pd.read_csv(os.path.join(DATA, "universe", "binance_um_top150.csv"))
    return u["symbol"].tolist()


def idx(tf):
    return pd.date_range(pd.Timestamp("2020-01-01", tz="UTC"), END, freq=TFS[tf], inclusive="left").astype("datetime64[ns, UTC]")


def main():
    S = syms()
    for tf, freq in TFS.items():
        ix = idx(tf)
        out = {f: {} for f in ["open", "high", "low", "close", "qv", "tbqv", "trades", "fund",
                               "oi_usd", "top_pos_ls", "top_acct_ls", "global_ls", "taker_ls_vol",
                               "prem", "prem_mean"]}
        if tf == "1d":
            out["spot_close"] = {}; out["spot_qv"] = {}
        delta = pd.Timedelta(freq)
        for n, s in enumerate(S):
            k = pd.read_parquet(os.path.join(DATA, "binance", "um", "klines", tf, f"{s}.parquet"))
            k["ts"] = k["ts"].astype("datetime64[ns, UTC]")
            k = k.set_index("ts")
            ok = k["n_bars"] >= 0.8 * FULL_MIN[tf]
            k = k.where(ok).reindex(ix)
            for f, c in [("open", "open"), ("high", "high"), ("low", "low"), ("close", "close"),
                         ("qv", "quote_volume"), ("tbqv", "taker_buy_quote_volume"), ("trades", "trades")]:
                out[f][s] = k[c].astype("float32")
            # funding attributed to the bar during which it is charged
            fp = os.path.join(DATA, "binance", "um", "funding", f"{s}.parquet")
            fs = pd.Series(np.nan, index=ix, dtype="float32")
            if os.path.exists(fp):
                f = pd.read_parquet(fp)[["ts", "funding_rate"]].dropna()
                t = f["ts"].astype("datetime64[ns, UTC]").dt.round("1min")
                bar = (t - pd.Timedelta("1ms")).dt.floor(freq)
                g = f.groupby(bar.values)["funding_rate"].sum()
                g.index = pd.DatetimeIndex(g.index).tz_convert("UTC") if g.index.tz is not None else pd.DatetimeIndex(g.index).tz_localize("UTC")
                fs = g.reindex(ix).astype("float32")
                # bars that are listed but had no settlement inside -> 0 funding
                fs = fs.where(fs.notna() | out["close"][s].isna(), 0.0)
            out["fund"][s] = fs
            # metrics as-of bar close
            mp = os.path.join(DATA, "binance", "um", "metrics", f"{s}.parquet")
            mcols = ["oi_usd", "top_pos_ls", "top_acct_ls", "global_ls", "taker_ls_vol"]
            if os.path.exists(mp):
                m = pd.read_parquet(mp, columns=["ts"] + mcols)
                m["ts"] = m["ts"].astype("datetime64[ns, UTC]")
                m = m.sort_values("ts").dropna(subset=["ts"])
                left = pd.DataFrame({"t": ix + delta})
                mm = pd.merge_asof(left, m.rename(columns={"ts": "t"}), on="t", direction="backward",
                                   tolerance=pd.Timedelta("2h"))
                for c in mcols:
                    out[c][s] = pd.Series(mm[c].values, index=ix, dtype="float32")
                del m, mm
            else:
                for c in mcols:
                    out[c][s] = pd.Series(np.nan, index=ix, dtype="float32")
            # premium index (1h bars, known at hour close)
            pp = os.path.join(DATA, "binance", "um", "premium_index", "1h", f"{s}.parquet")
            if os.path.exists(pp):
                p = pd.read_parquet(pp, columns=["ts", "prem_close"])
                p["ts"] = p["ts"].astype("datetime64[ns, UTC]")
                p = p.set_index("ts")["prem_close"]
                if tf == "1h":
                    out["prem"][s] = p.reindex(ix).astype("float32")
                    out["prem_mean"][s] = out["prem"][s]
                else:
                    b = p.index.floor(freq)
                    out["prem"][s] = p.groupby(b).last().reindex(ix).astype("float32")
                    out["prem_mean"][s] = p.groupby(b).mean().reindex(ix).astype("float32")
            else:
                out["prem"][s] = pd.Series(np.nan, index=ix, dtype="float32")
                out["prem_mean"][s] = pd.Series(np.nan, index=ix, dtype="float32")
            if tf == "1d":
                sp = os.path.join(DATA, "binance", "spot", "klines", "1d", f"{s}.parquet")
                if os.path.exists(sp):
                    q = pd.read_parquet(sp)
                    q["ts"] = q["ts"].astype("datetime64[ns, UTC]")
                    q = q.set_index("ts")
                    q = q.where(q["n_bars"] >= 0.8 * 24).reindex(ix)  # spot 1d resampled from 1h -> n_bars in hours
                    out["spot_close"][s] = q["close"].astype("float32")
                    out["spot_qv"][s] = q["quote_volume"].astype("float32")
                else:
                    out["spot_close"][s] = pd.Series(np.nan, index=ix, dtype="float32")
                    out["spot_qv"][s] = pd.Series(np.nan, index=ix, dtype="float32")
            if n % 25 == 0:
                print(tf, n, s, flush=True)
        os.makedirs(os.path.join(CACHE, tf), exist_ok=True)
        for f, d in out.items():
            df = pd.DataFrame(d, index=ix).astype("float32")
            df.index.name = "ts"
            df.to_parquet(os.path.join(CACHE, tf, f"{f}.parquet"), compression="zstd")
        print("done", tf, flush=True)
        del out


if __name__ == "__main__":
    main()
