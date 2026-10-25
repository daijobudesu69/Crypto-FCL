import json, urllib.request, pandas as pd, numpy as np, time
D="../../data/hyperliquid"
def post(body):
    r=urllib.request.Request("https://api.hyperliquid.xyz/info", data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(r,timeout=30).read())
coins=["BTC","ETH","SOL","DOGE","kPEPE","WIF","TIA","SUI","ENA","ZEC"]
out=[]
for c in coins:
    for (a,b) in [("2025-03-01","2025-03-10"),("2026-06-01","2026-06-10")]:
        s=int(pd.Timestamp(a,tz="UTC").timestamp()*1000); e=int(pd.Timestamp(b,tz="UTC").timestamp()*1000)
        try: fh=post({"type":"fundingHistory","coin":c,"startTime":s,"endTime":e})
        except Exception as ex: print(c,ex); continue
        api=pd.Series({pd.Timestamp(x["time"],unit="ms",tz="UTC").round("1min"):float(x["fundingRate"]) for x in fh})
        L=pd.read_parquet(f"{D}/funding/{c}.parquet"); L["ts"]=pd.to_datetime(L["ts"],utc=True).dt.round("1min")
        mine=L.set_index("ts")["funding_rate"].reindex(api.index)
        out.append(dict(coin=c,win=a,kind="funding",n=len(api),missing=int(mine.isna().sum()),maxdiff=float((mine-api).abs().max())))
        time.sleep(0.3)
    cs=post({"type":"candleSnapshot","req":{"coin":c,"interval":"1d","startTime":int(pd.Timestamp("2025-01-01",tz="UTC").timestamp()*1000),"endTime":int(pd.Timestamp("2026-10-02",tz="UTC").timestamp()*1000)}})
    api=pd.Series({pd.Timestamp(x["t"],unit="ms",tz="UTC"):float(x["c"]) for x in cs})
    K=pd.read_parquet(f"{D}/candles/1d/{c}.parquet"); K["ts"]=pd.to_datetime(K["ts"],utc=True)
    mine=K.set_index("ts")["close"].reindex(api.index)
    out.append(dict(coin=c,win="2025-01..2026-10",kind="close1d",n=len(api),missing=int(mine.isna().sum()),maxdiff=float(((mine-api)/api).abs().max())))
    print(out[-3:],flush=True); time.sleep(0.3)
pd.DataFrame(out).to_csv("integrity_hype.csv",index=False)
print(pd.DataFrame(out).to_string())
