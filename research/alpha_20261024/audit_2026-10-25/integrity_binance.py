import io, zipfile, urllib.request, pandas as pd, numpy as np, json, sys
D = "../../data/binance/um"
syms = ["BTCUSDT","ETHUSDT","SOLUSDT","DOGEUSDT","XRPUSDT","1000PEPEUSDT","WIFUSDT","TIAUSDT","SUIUSDT","ENAUSDT"]
months = ["2021-05","2023-06","2025-02","2025-10","2026-06"]
def get(url):
    try:
        z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url, timeout=30).read()))
        return pd.read_csv(z.open(z.namelist()[0]), header=None)
    except Exception as e:
        return None
res=[]
for s in syms:
    F = pd.read_parquet(f"{D}/funding/{s}.parquet"); F["ts"]=pd.to_datetime(F["ts"],utc=True).dt.round("1min")
    K = pd.read_parquet(f"{D}/klines/1d/{s}.parquet"); K["ts"]=pd.to_datetime(K["ts"],utc=True)
    for m in months:
        f = get(f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip")
        k = get(f"https://data.binance.vision/data/futures/um/monthly/klines/{s}/1d/{s}-1d-{m}.zip")
        row={"sym":s,"month":m}
        if f is not None:
            if not str(f.iloc[0,0]).isdigit(): f=f.iloc[1:]
            f.columns=["t","ih","rate"][:f.shape[1]] if f.shape[1]==3 else list(range(f.shape[1]))
            ft = pd.to_datetime(f.iloc[:,0].astype("int64"),unit="ms",utc=True).dt.round("1min")
            fr = pd.Series(f.iloc[:,-1].astype(float).values, index=ft)
            mine = F.set_index("ts")["funding_rate"].reindex(fr.index)
            row.update(f_n=len(fr), f_missing=int(mine.isna().sum()), f_maxabsdiff=float((mine-fr).abs().max()))
            lake_m = F[(F.ts>=fr.index.min())&(F.ts<=fr.index.max())]
            row["f_extra_in_lake"]=int(len(lake_m)-len(fr))
        else: row["f_n"]=None
        if k is not None:
            if not str(k.iloc[0,0]).isdigit(): k=k.iloc[1:]
            kt = pd.to_datetime(k.iloc[:,0].astype("int64"),unit="ms",utc=True)
            kc = pd.Series(k.iloc[:,4].astype(float).values, index=kt)
            kq = pd.Series(k.iloc[:,7].astype(float).values, index=kt)
            mc = K.set_index("ts")["close"].reindex(kt); mq=K.set_index("ts")["quote_volume"].reindex(kt)
            row.update(k_n=len(kc), k_missing=int(mc.isna().sum()), k_close_maxrel=float(((mc-kc)/kc).abs().max()), k_qv_maxrel=float(((mq-kq)/kq).abs().max()))
        else: row["k_n"]=None
        res.append(row); print(row, flush=True)
pd.DataFrame(res).to_csv("integrity_binance.csv", index=False)
