"""Independent panel builder from RAW parquet (no cache, no lab.py). Saves npz-ish pickles."""
import pandas as pd, numpy as np, os, json, urllib.request
D="../../data"
syms=pd.read_csv(f"{D}/universe/binance_um_top150.csv")["symbol"].tolist()
ix=pd.date_range("2020-01-01","2026-10-02",freq="1D",tz="UTC")
hix=pd.date_range("2020-01-01","2026-10-02 23:00",freq="1h",tz="UTC")
# independent HYPE map: from HYPE meta, Binance sym -> HYPE name (strip USDT, handle 1000x/k prefix)
meta=json.loads(urllib.request.urlopen(urllib.request.Request("https://api.hyperliquid.xyz/info",data=b'{"type":"meta"}',headers={"Content-Type":"application/json"})).read())
hl={u["name"] for u in meta["universe"] if not u.get("isDelisted")}
hl_all={u["name"] for u in meta["universe"]}
def tohl(s):
    b=s[:-4]
    for cand in [b, "k"+b[4:] if b.startswith("1000") else None, b[4:] if b.startswith("1000") else None, "k"+b[7:] if b.startswith("1000000") else None]:
        if cand and cand in hl_all: return cand
    return None
mp={s:tohl(s) for s in syms}
C={};Q={};F={};H={};HC={};HF={};HQ={};NB={}
for s in syms:
    k=pd.read_parquet(f"{D}/binance/um/klines/1d/{s}.parquet"); k["ts"]=pd.to_datetime(k["ts"],utc=True); k=k.set_index("ts").reindex(ix)
    C[s]=k["close"]; Q[s]=k["quote_volume"]; NB[s]=k["n_bars"]
    f=pd.read_parquet(f"{D}/binance/um/funding/{s}.parquet"); t=pd.to_datetime(f["ts"],utc=True).dt.round("1min")
    # settlement at time T belongs to day whose window (open, open+24h] contains T
    day=(t-pd.Timedelta("1s")).dt.floor("1D")
    F[s]=f.groupby(day)["funding_rate"].sum().reindex(ix)
    h=pd.read_parquet(f"{D}/binance/um/klines/1h/{s}.parquet",columns=["ts","close"]); h["ts"]=pd.to_datetime(h["ts"],utc=True)
    H[s]=h.set_index("ts")["close"].reindex(hix)
    n=mp[s]
    if n and os.path.exists(f"{D}/hyperliquid/candles/1d/{n}.parquet"):
        c=pd.read_parquet(f"{D}/hyperliquid/candles/1d/{n}.parquet"); c["ts"]=pd.to_datetime(c["ts"],utc=True); c=c.set_index("ts").reindex(ix)
        HC[s]=c["close"]; HQ[s]=c["quote_volume"]
        g=pd.read_parquet(f"{D}/hyperliquid/funding/{n}.parquet"); t=pd.to_datetime(g["ts"],utc=True).dt.round("1min")
        HF[s]=g.groupby((t-pd.Timedelta("1s")).dt.floor("1D"))["funding_rate"].sum().reindex(ix)
P={k:pd.DataFrame(v) for k,v in dict(C=C,Q=Q,F=F,NB=NB,HC=HC,HF=HF,HQ=HQ).items()}
P["H"]=pd.DataFrame(H).astype("float32")
pd.to_pickle(P,"indep_panels.pkl"); pd.Series(mp).to_csv("indep_hypemap.csv")
am=pd.read_csv("../hype_map.csv")
cmp=am.set_index("binance")["hype"].reindex(syms)
mine=pd.Series(mp)
print("author mapped",cmp.notna().sum(),"mine",mine.notna().sum(),"disagree",[(s,cmp[s],mine[s]) for s in syms if str(cmp[s])!=str(mine[s])])
print("delisted on HL among mine:",[mine[s] for s in syms if mine[s] and mine[s] not in hl])
