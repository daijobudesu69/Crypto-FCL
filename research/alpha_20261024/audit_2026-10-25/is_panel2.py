import os, pandas as pd, numpy as np
ix=pd.date_range("2020-01-01","2024-12-31",freq="1D",tz="UTC")
lake=[l.split(',')[1] for l in open('../../data/universe/binance_um_top150.csv').read().strip().split('\n')[1:]]
P=pd.read_pickle("indep_panels.pkl")
C={s:P["C"][s].where(P["NB"][s]>=0.8*1440).reindex(ix) for s in lake}
Q={s:P["Q"][s].reindex(ix) for s in lake}; F={s:P["F"][s].reindex(ix) for s in lake}
for fn in os.listdir("is_sym"):
    s=fn[:-4]
    if s in C: continue
    d=pd.read_pickle(f"is_sym/{fn}"); K=d["k"]; FF=d["f"]
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
print("symbols",c.shape[1],"died/delisted before 2025:",len(dead)); print(sorted(dead.index))
