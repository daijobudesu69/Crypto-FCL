from indep_sim import *
import numpy as np, pandas as pd
def legs(start,end,venue):
    # long-only and short-only books (each 0.5 gross) using same selection, plus EW universe
    d,_,_=run(start=start,end=end,venue=venue)
    return d
C=P["C"]; btc=C["BTCUSDT"].pct_change()
for nm,st,en,ven in [("FWD hype","2025-01-01",None,"hype"),("IS binance","2020-03-01","2024-12-31","binance")]:
    d=legs(st,en,ven); r=d["r"]
    syms=[s for s in C.columns if isinstance(MAP.get(s),str)]
    ew=C[syms].pct_change().where(C[syms].shift(1).notna()).mean(axis=1)
    X=pd.DataFrame({"r":r,"btc":btc.reindex(r.index),"ew":ew.reindex(r.index)}).dropna()
    b=np.linalg.lstsq(np.c_[np.ones(len(X)),X[["btc","ew"]].values],X["r"].values,rcond=None)[0]
    print(nm,"corr btc",round(X.r.corr(X.btc),3),"corr ewalt",round(X.r.corr(X.ew),3),"beta btc/ew",b[1:].round(3),"alpha/yr",round(b[0]*365,3))
    print("  avg net exposure (long-short)/eq",round(d["net"].mean(),3), "avg nl/ns",d.nl.mean().round(1),d.ns.mean().round(1))
    # yearly
    y=(1+r).groupby(r.index.year).prod()-1; sh=r.groupby(r.index.year).apply(lambda x:x.mean()/x.std()*np.sqrt(365))
    print("  yearly ret",y.round(3).to_dict(),"sharpe",sh.round(2).to_dict())
    # performance by market regime: ew alt 30d trailing return up vs down
    reg=(1+ew).rolling(30).apply(np.prod,raw=True).shift(1).reindex(r.index)-1
    up=r[reg>0]; dn=r[reg<=0]
    print("  alt 30d UP days",len(up),"ann ret",round(up.mean()*365,3),"SR",round(up.mean()/up.std()*np.sqrt(365),2),
          "| DOWN days",len(dn),"ann ret",round(dn.mean()*365,3),"SR",round(dn.mean()/dn.std()*np.sqrt(365),2))
    bt=(1+btc).rolling(30).apply(np.prod,raw=True).shift(1).reindex(r.index)-1
    up=r[bt>0]; dn=r[bt<=0]
    print("  btc 30d UP ann",round(up.mean()*365,3),"SR",round(up.mean()/up.std()*np.sqrt(365),2),"| DOWN ann",round(dn.mean()*365,3),"SR",round(dn.mean()/dn.std()*np.sqrt(365),2))
    # best days concentration
    lr=np.log1p(r); srt=lr.sort_values(ascending=False)
    print("  log-ret total",round(lr.sum(),3),"top10 days share",round(srt.head(10).sum()/lr.sum(),3),"top20",round(srt.head(20).sum()/lr.sum(),3))
    # pnl split
    print("  PnL harga $",round(d.pnl.sum(),1)," funding $",round(d.fund.sum(),1)," fee $",round(-d.fee.sum(),1))
