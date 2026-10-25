import urllib.request, re, json
from concurrent.futures import ThreadPoolExecutor
B="https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix="
def ls(prefix, marker=""):
    out=[]; 
    while True:
        x=urllib.request.urlopen(B+prefix+("&marker="+marker if marker else ""),timeout=60).read().decode()
        out+=re.findall(r"<Prefix>([^<]+)</Prefix>",x)[1:] + re.findall(r"<Key>([^<]+)</Key>",x)
        if "<IsTruncated>true</IsTruncated>" not in x: break
        marker=re.findall(r"<NextMarker>([^<]+)</NextMarker>",x)[0]
    return out
syms=[p.split("/")[-2] for p in ls("data/futures/um/monthly/klines/")]
syms=[s for s in syms if s.endswith("USDT")]
def months(s):
    try:
        k=ls(f"data/futures/um/monthly/klines/{s}/1d/")
        ms=sorted(set(re.findall(r"-(\d{4}-\d{2})\.zip$"," ".join(k).replace(" ","\n"),re.M))) if k else []
        ms=sorted(set(m for f in k for m in re.findall(r"(\d{4}-\d{2})\.zip$",f)))
        return s,ms
    except Exception as e: return s,None
with ThreadPoolExecutor(24) as ex: res=dict(ex.map(months,syms))
json.dump(res,open("binance_um_months.json","w"))
alive_aug24=[s for s,m in res.items() if m and m[0]<="2024-08" and m[-1]>="2024-08"]
listed_after=[s for s,m in res.items() if m and m[0]>"2024-08" and m[0]<="2026-07"]
inwin=[s for s,m in res.items() if m and m[-1]>="2024-08" and m[0]<="2026-07"]
dead=[s for s in inwin if res[s][-1]<"2026-07"]
print("USDT perps total",len(syms),"traded in window",len(inwin),"alive Aug24",len(alive_aug24),"listed in window",len(listed_after))
print("stopped before 2026-07 (delisted/renamed in window):",len(dead))
print(sorted(dead)[:200])
