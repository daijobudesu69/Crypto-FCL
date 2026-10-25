import pandas as pd, numpy as np, json
P=pd.read_pickle("is_full_panels.pkl")
lake=[l.split(',')[1] for l in open('../../data/universe/binance_um_top150.csv').read().strip().split('\n')[1:]]
MAP=pd.read_csv("indep_hypemap.csv",index_col=0).iloc[:,0]; hype={s for s,v in MAP.items() if isinstance(v,str)}
Call=P["C"]; last=Call.apply(lambda x:x.last_valid_index()); dead=set(last[last<Call.index[-1]-pd.Timedelta("3D")].index)
def run(rank_set, keep_set, k_mode="cap", stop=0.40, momskip=0.9, cost=0.00065, eq0=200.0, start="2020-03-01"):
    rs=[s for s in Call.columns if s in rank_set]
    C=Call[rs]; Q=P["Q"][rs]; F=P["F"].reindex(columns=rs)
    hist=C.notna().cumsum(); elig=C.notna()&(hist>=60)
    adv=Q.rolling(30,min_periods=15).median(); rk=adv.where(elig).rank(axis=1,ascending=False)
    keep=pd.Series([s in keep_set for s in rs],index=rs)
    U=(elig&(rk<=50)&keep).values
    S=F.where(C.notna()).fillna(0).rolling(14,min_periods=14).sum().where(hist>=14).values
    mo=(C/C.shift(14)-1).values; c=C.values; fv=np.nan_to_num(F.values)
    ix=C.index; t0=ix.get_loc(pd.Timestamp(start,tz="UTC")); m=len(rs)
    pos=np.zeros(m); entry=np.full(m,np.nan); ban=np.zeros(m,bool); eq=eq0; rows=[]; dexp=0; texp=0
    isd=np.array([s in dead for s in rs]); dpnl=0
    for t in range(t0,len(ix)-1):
        ok=U[t]&~np.isnan(S[t])&~np.isnan(c[t]); ids=np.flatnonzero(ok); N=len(ids); tgt=np.zeros(m)
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
        r=c[t+1]/c[t]-1; r=np.where(np.isnan(r),0.0,r)
        hit=np.zeros(m,bool)
        if stop:   # daily-close stop (no hourly data for dead coins)
            with np.errstate(invalid="ignore"): hit=(new<0)&(c[t+1]>=entry*(1+stop))
        pnl=new*r; fund=-(new*fv[t+1])
        sfee=(np.abs(new[hit])*(1+r[hit])).sum()*cost
        dexp+=np.abs(new[isd]).sum(); texp+=np.abs(new).sum(); dpnl+=(pnl+fund)[isd].sum()
        eq+=pnl.sum()+fund.sum()-fee-sfee; rows.append((ix[t+1],eq))
        pos=new*(1+r); pos[hit]=0; ban|=hit; pos[np.isnan(c[t+1])]=0
        if eq<=0: break
    d=pd.DataFrame(rows,columns=["ts","eq"]).set_index("ts"); r=d["eq"].pct_change().fillna(d["eq"].iloc[0]/eq0-1)
    sub={}
    for nm,a,b in [("20-21","2020","2022"),("22","2022","2023"),("23-24","2023","2025")]:
        x=r[(r.index>=pd.Timestamp(a,tz="UTC"))&(r.index<pd.Timestamp(b,tz="UTC"))]; sub[nm]=round(x.mean()/x.std()*np.sqrt(365),2)
    return dict(end=round(d["eq"].iloc[-1],0),sharpe=round(r.mean()/r.std()*np.sqrt(365),2),maxdd=round((d["eq"]/d["eq"].cummax()-1).min(),3),
                sub=sub,dead_share=round(dexp/texp,3),dead_pnl=round(dpnl,1))
if __name__=="__main__":
    allc=set(Call.columns); L=set(lake)
    out={}
    for nm,rs,ks in [("A: author (rank lake150, keep HYPE)",L,hype),("B: lake150, no HYPE filter",L,L),("C: ALL 451 incl dead",allc,allc),("D: ALL ranked, keep HYPE",allc,hype)]:
        for km in ["cap","ter"]:
            x=run(rs,ks,k_mode=km); out[f"{nm}|{km}"]=x; print(f"{nm:38s}{km}",x,flush=True)
    json.dump(out,open("is_surv_results.json","w"),indent=1)
