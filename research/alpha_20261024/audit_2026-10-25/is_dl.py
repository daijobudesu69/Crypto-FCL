import json, os, io, zipfile, urllib.request, time, pandas as pd
from concurrent.futures import ThreadPoolExecutor
T0=time.time()
r=json.load(open('binance_um_months.json'))
lake={l.split(',')[1] for l in open('../../data/universe/binance_um_top150.csv').read().strip().split('\n')[1:]}
extra=sorted(s for s,m in r.items() if m and m[0]<='2024-12' and m[-1]>='2020-01' and s not in lake)
def get(u):
    for a in range(3):
        try:
            z=zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(u,timeout=30).read()))
            d=pd.read_csv(z.open(z.namelist()[0]),header=None)
            if not str(d.iloc[0,0]).lstrip('-').isdigit(): d=d.iloc[1:]
            return d
        except urllib.error.HTTPError as e:
            if e.code==404: return None
        except Exception: time.sleep(1)
    return None
def one(s):
    fn=f"is_extra/{s}.pkl"
    if os.path.exists(fn): return
    ms=[m for m in r[s] if '2019-12'<=m<='2024-12']
    k=[get(f"https://data.binance.vision/data/futures/um/monthly/klines/{s}/1d/{s}-1d-{m}.zip") for m in ms]
    f=[get(f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip") for m in ms]
    k=[x for x in k if x is not None]; f=[x for x in f if x is not None]
    K=pd.concat(k) if k else None; F=pd.concat(f) if f else None
    pd.to_pickle({"k":K,"f":F},fn)
todo=[s for s in extra if not os.path.exists(f"is_extra/{s}.pkl")]
with ThreadPoolExecutor(32) as ex:
    futs=[]
    for s in todo:
        futs.append(ex.submit(one,s))
    for x in futs:
        if time.time()-T0>165: break
        try: x.result(timeout=max(1,165-(time.time()-T0)))
        except Exception: pass
    ex.shutdown(wait=False,cancel_futures=True)
print("have",len(os.listdir("is_extra")),"of",len(extra))
os._exit(0)
