import json, os, io, zipfile, urllib.request, time
from concurrent.futures import ThreadPoolExecutor, wait
T0=time.time()
r=json.load(open('binance_um_months.json'))
lake={l.split(',')[1] for l in open('../../data/universe/binance_um_top150.csv').read().strip().split('\n')[1:]}
extra=sorted(s for s,m in r.items() if m and m[0]<='2024-12' and m[-1]>='2020-01' and s not in lake)
done_sym={f[:-4] for f in os.listdir("is_extra")}
have=set(os.listdir("is_raw"))
tasks=[]
for s in extra:
    if s in done_sym: continue
    for m in r[s]:
        if '2019-12'<=m<='2024-12':
            for kind,u in [("k",f"https://data.binance.vision/data/futures/um/monthly/klines/{s}/1d/{s}-1d-{m}.zip"),("f",f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip")]:
                fn=f"{s}__{kind}__{m}.csv"
                if fn not in have and fn+".404" not in have: tasks.append((u,fn))
def get(t):
    u,fn=t
    try:
        b=urllib.request.urlopen(u,timeout=20).read()
        z=zipfile.ZipFile(io.BytesIO(b)); open(f"is_raw/{fn}","wb").write(z.read(z.namelist()[0]))
    except urllib.error.HTTPError as e:
        if e.code==404: open(f"is_raw/{fn}.404","w").close()
    except Exception: pass
ex=ThreadPoolExecutor(96); i=0
while i<len(tasks) and time.time()-T0<120:
    fs=[ex.submit(get,t) for t in tasks[i:i+800]]; wait(fs,timeout=30); i+=800
print("tasks",len(tasks),"processed",min(i,len(tasks)),"files now",len(os.listdir("is_raw")))
os._exit(0)
