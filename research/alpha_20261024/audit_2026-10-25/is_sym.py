import os, time, pandas as pd, numpy as np
T0=time.time()
def rd(fp):
    if os.path.getsize(fp)==0: return None
    d=pd.read_csv(fp,header=None)
    if not str(d.iloc[0,0]).lstrip('-').isdigit(): d=d.iloc[1:]
    return d
bys={}
for f in os.listdir("is_raw"):
    if f.endswith(".404"): continue
    s,k,m=f[:-4].split("__"); bys.setdefault(s,{}).setdefault(k,[]).append(f)
for s in {f[:-4] for f in os.listdir("is_extra")}:
    d=pd.read_pickle(f"is_extra/{s}.pkl"); pd.to_pickle({"k":d["k"],"f":d["f"]},f"is_sym/{s}.pkl")
done=set(os.listdir("is_sym"))
for s,kk in bys.items():
    if s+".pkl" in done: continue
    if time.time()-T0>140: break
    K=[x for x in (rd(f"is_raw/{f}") for f in kk.get("k",[])) if x is not None]
    F=[x for x in (rd(f"is_raw/{f}") for f in kk.get("f",[])) if x is not None]
    pd.to_pickle({"k":pd.concat(K) if K else None,"f":pd.concat(F) if F else None},f"is_sym/{s}.pkl")
print(len(os.listdir("is_sym")),"of",len(bys)+len(os.listdir("is_extra")))
