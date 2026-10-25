import pandas as pd, numpy as np, json
P=pd.read_pickle("surv_panels.pkl")
alive150=set(pd.read_csv("../../data/universe/binance_um_top150.csv")["symbol"])
MAP=pd.read_csv("indep_hypemap.csv",index_col=0).iloc[:,0]; hype={s for s,v in MAP.items() if isinstance(v,str)}
Call=P["C"]; last=Call.apply(lambda x:x.last_valid_index()); dead=set(last[last<Call.index[-1]-pd.Timedelta("3D")].index)
def run(allowed,start="2024-09-01",k_mode="cap",stop=0.40,momskip=0.9,need_next=False,cost=0.00065,eq0=200.0):
    syms=[s for s in Call.columns if s in allowed]
    C=Call[syms]; Q=P["Q"][syms]; F=P["F"].reindex(columns=syms)
    hist=C.notna().cumsum(); elig=C.notna()&(hist>=60)
    adv=Q.rolling(30,min_periods=15).median(); rk=adv.where(elig).rank(axis=1,ascending=False)
    U=(elig&(rk<=50)).values
    Fz=F.where(C.notna()).fillna(0.0).where(C.notna()|F.notna())
    S=F.where(C.notna()).fillna(0).rolling(14,min_periods=14).sum().where(hist>=14).values
    mo=(C/C.shift(14)-1).values; c=C.values; fv=np.nan_to_num(F.values)
    H=P["H"][syms].values.reshape(len(C),24,-1)
    ix=C.index; t0=ix.get_loc(pd.Timestamp(start,tz="UTC"))
    m=len(syms); pos=np.zeros(m); entry=np.full(m,np.nan); ban=np.zeros(m,bool); eq=eq0; rows=[]; deadexp=0; totexp=0
    for t in range(t0,len(ix)-1):
        ok=U[t]&~np.isnan(S[t])&~np.isnan(c[t])
        if need_next: ok&=~np.isnan(c[t+1])
        ids=np.flatnonzero(ok); N=len(ids); tgt=np.zeros(m)
        if N>=12:
            K=N//3 if k_mode=="ter" else int(min(N//3,max(4,np.floor(eq/2/12.5))))
            mr=pd.Series(mo[t,ids]).rank(pct=True).values
            cs=[ids[i] for i in np.argsort(-S[t,ids],kind="stable") if (np.isnan(mr[i]) or mr[i]<=momskip)]
            sh=[j for j in cs if not ban[j]][:K]; lg=[j for j in ids[np.argsort(S[t,ids],kind="stable")] if j not in sh][:K]
            if sh and lg: tgt[sh]=-0.5/len(sh); tgt[lg]=0.5/len(lg)
        ban[:]=False; tu=tgt*eq
        if k_mode=="cap":
            tu[(np.abs(tu)<10)&(tu!=0)]=0; d=tu-pos
            tr=(np.abs(d)>=10)|((tu==0)&(pos!=0))|((pos==0)&(tu!=0)); new=np.where(tr,tu,pos)
        else: new=tu
        fee=np.abs(new-pos).sum()*cost
        ns=(new<0)&~(pos<0); entry[ns]=c[t,ns]; entry[new>=0]=np.nan
        r=c[t+1]/c[t]-1; r=np.where(np.isnan(r),0.0,r)   # delisted next day -> exit at last traded close
        hit=np.zeros(m,bool)
        if stop:
            hc=H[t+1]; lvl=entry*(1+stop)
            with np.errstate(invalid="ignore"): cr=(hc>=lvl[None,:])&(new<0)[None,:]
            hit=cr.any(0)
            if hit.any():
                fi=np.argmax(cr,0); fill=hc[fi,np.arange(m)]*1.005; r=r.copy(); r[hit]=fill[hit]/c[t,hit]-1
        pnl=(new*r).sum(); fund=-(new*fv[t+1]*np.where(hit,0.5,1)).sum(); sfee=(np.abs(new[hit])*(1+r[hit])).sum()*cost
        isd=np.array([s in dead for s in syms]); deadexp+=np.abs(new[isd]).sum(); totexp+=np.abs(new).sum()
        eq+=pnl+fund-fee-sfee; rows.append((ix[t+1],eq))
        pos=new*(1+r); pos[hit]=0; ban|=hit
        pos[np.isnan(c[t+1])]=0  # delisted: position settled
    d=pd.DataFrame(rows,columns=["ts","eq"]).set_index("ts"); r=d["eq"].pct_change().fillna(d["eq"].iloc[0]/eq0-1)
    return dict(end=round(d["eq"].iloc[-1],1),sharpe=round(r.mean()/r.std()*np.sqrt(365),2),maxdd=round((d["eq"]/d["eq"].cummax()-1).min(),3),
                dead_share=round(deadexp/totexp,4),yrs={int(k):round(v,3) for k,v in ((1+r).groupby(r.index.year).prod()-1).items()})
if __name__=="__main__":
    allc=set(Call.columns); alive_end=allc-dead
    out={}
    for nm,al in [("ALL incl delisted",allc),("survivors only (alive at end)",alive_end),("lake 150 only",alive150),("HYPE-listed only",hype)]:
        for km in ["cap","ter"]:
            x=run(al,k_mode=km); out[f"{nm}|{km}"]=x; print(f"{nm:32s} {km}",x,flush=True)
    json.dump(out,open("surv_results.json","w"),indent=1)
