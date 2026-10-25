"""Shift the daily decision time to HH:00 UTC using 1h Binance closes + raw settlements."""
import pandas as pd, numpy as np, sys
h=int(sys.argv[1]); D="../../data"
P=pd.read_pickle("indep_panels.pkl")
syms=list(P["C"].columns)
ix=P["C"].index; H=P["H"]
# daily close at d+h+24h  == close of 1h bar opening at d+h+23h
C2=H.reindex(ix+pd.Timedelta(hours=h+23)); C2.index=ix
F2={}
for s in syms:
    f=pd.read_parquet(f"{D}/binance/um/funding/{s}.parquet"); t=pd.to_datetime(f["ts"],utc=True).dt.round("1min")
    F2[s]=f.groupby((t-pd.Timedelta(hours=h)-pd.Timedelta("1s")).dt.floor("1D"))["funding_rate"].sum().reindex(ix)
H2=H.shift(-h)   # hour-slot j of day d = hour d+h+j
P2=dict(P); P2["C"]=C2.astype(float); P2["F"]=pd.DataFrame(F2); P2["H"]=H2
P2["NB"]=pd.DataFrame(1440,index=ix,columns=syms).where(C2.notna())
pd.to_pickle(P2,f"indep_panels_h{h}.pkl")
