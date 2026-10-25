import os, zipfile, io, time, glob, pandas as pd, numpy as np, sys
T0=time.time(); R=os.path.expanduser("~/mnt/Crypto data 2/data/raw")
syms=sorted(os.listdir(f"{R}/klines"))
part=int(sys.argv[1]) if len(sys.argv)>1 else 0; syms=syms[part::2]
done=set(f[:-4] for f in os.listdir("surv"))
def rd(fp):
    z=zipfile.ZipFile(fp); d=pd.read_csv(z.open(z.namelist()[0]),header=None)
    if not str(d.iloc[0,0]).lstrip("-").isdigit(): d=d.iloc[1:]
    return d
for s in syms:
    if s in done: continue
    if time.time()-T0>160: break
    ks=[rd(f) for f in sorted(glob.glob(f"{R}/klines/{s}/*.zip"))]
    if not ks: open(f"surv/{s}.pkl","wb").close(); continue
    k=pd.concat(ks); ts=pd.to_datetime(k[0].astype("int64"),unit="ms",utc=True)
    h=pd.DataFrame({"close":k[4].astype(float).values,"qv":k[7].astype(float).values},index=ts).sort_index()
    h=h[~h.index.duplicated()]
    fs=[rd(f) for f in sorted(glob.glob(f"{R}/fundingRate/{s}/*.zip"))]
    if fs:
        f=pd.concat(fs); ft=pd.to_datetime(f[0].astype("int64"),unit="ms",utc=True).dt.round("1min")
        fr=pd.Series(f.iloc[:,-1].astype(float).values,index=ft); fr=fr[~fr.index.duplicated()]
    else: fr=pd.Series(dtype=float)
    pd.to_pickle({"h":h,"f":fr},f"surv/{s}.pkl")
print("done",len(os.listdir("surv")),"of",len(syms))
