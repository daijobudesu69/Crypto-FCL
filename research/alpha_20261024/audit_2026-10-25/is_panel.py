import os, glob, pandas as pd, numpy as np, json
ix=pd.date_range("2020-01-01","2024-12-31",freq="1D",tz="UTC")
lake=[l.split(',')[1] for l in open('../../data/universe/binance_um_top150.csv').read().strip().split('\n')[1:]]
C={};Q={};F={}
P=pd.read_pickle("indep_panels.pkl")
for s in lake:
    C[s]=P["C"][s].where(P["NB"][s]>=0.8*1440).reindex(ix); Q[s]=P["Q"][s].reindex(ix); F[s]=P["F"][s].reindex(ix)
def rd(fp):
    if os.path.getsize(fp)==0: return None
    d=pd.read_csv(fp,header=None)
    if not str(d.iloc[0,0]).lstrip('-').isdigit(): d=d.iloc[1:]
    return d
files=os.listdir("is_raw"); bys={}
for f in files:
    if f.endswith(".404"): continue
    s,k,m=f[:-4].split("__"); bys.setdefault((s,k),[]).append(f)
for s in {f[:-4] for f in os.listdir("is_extra")}:
    d=pd.read_pickle(f"is_extra/{s}.pkl")
    bys[(s,"kpk")]=d
syms=sorted({s for s,k in bys})
for s in syms:
    if s in C: continue
    if (s,"kpk") in bys:
        d=bys[(s,"kpk")]; K=d["k"]; FF=d["f"]
    else:
        K=pd.concat([x for x in [rd(f"is_raw/{f}") for f in bys.get((s,"k"),[])] if x is not None] or [pd.DataFrame()])
        FF=pd.concat([x for x in [rd(f"is_raw/{f}") for f in bys.get((s,"f"),[])] if x is not None] or [pd.DataFrame()])
    if K is None or len(K)==0: continue
    t=pd.to_datetime(K[0].astype("int64"),unit="ms",utc=True)
    k=pd.DataFrame({"c":K[4].astype(float).values,"q":K[7].astype(float).values},index=t); k=k[~k.index.duplicated()].sort_index().reindex(ix)
    C[s]=k["c"].where(k["q"]>0); Q[s]=k["q"].where(k["q"]>0)
    if FF is not None and len(FF):
        ft=pd.to_datetime(FF[0].astype("int64"),unit="ms",utc=True).dt.round("1min"); fr=pd.Series(FF.iloc[:,-1].astype(float).values,index=ft)
        fr=fr[~fr.index.duplicated()]; F[s]=fr.groupby((fr.index-pd.Timedelta("1s")).floor("1D")).sum().reindex(ix)
out={"C":pd.DataFrame(C),"Q":pd.DataFrame(Q),"F":pd.DataFrame(F)}
pd.to_pickle(out,"is_full_panels.pkl")
c=out["C"]; last=c.apply(lambda x:x.last_valid_index())
dead=last[last<ix[-1]-pd.Timedelta("3D")]
print("symbols",c.shape[1],"lake",len(lake),"died/delisted before 2025:",len(dead))
print(sorted(dead.index)[:60])
