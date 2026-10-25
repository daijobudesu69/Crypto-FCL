"""Independent FCL v1.0 replay written from the spec text (not from fcl.py). Inputs: indep_panels.pkl."""
import pandas as pd, numpy as np
P=pd.read_pickle("indep_panels.pkl")
MAP=pd.read_csv("indep_hypemap.csv",index_col=0).iloc[:,0]
def run(start,end=None,venue="hype",eq0=200.0,cost=0.00065,stop=0.40,momskip=0.90,k_mode="cap",lag=0,
        partial_nan=True,need_next=True,top_k=50,rank_within_hype=False,exclude=(),fee_mult=1.0,fund_on=True,
        drop_top=None, signal="F14", L=14, perm=None, keep=False):
    C=P["C"].copy(); Q=P["Q"]; F=P["F"]
    if partial_nan: C=C.where(P["NB"]>=0.8*1440)
    syms=[s for s in C.columns if isinstance(MAP.get(s),str) and s not in exclude]
    if venue=="hype":
        syms=[s for s in syms if s in P["HC"].columns]
    C=C[syms]; Q=Q[syms].where(C.notna()); F=F[syms]
    hist=C.notna().cumsum()
    elig=C.notna()&(hist>=60)
    adv=(Q).rolling(30,min_periods=15).median()
    allC=P["C"].where(P["NB"]>=0.8*1440) if partial_nan else P["C"]
    if rank_within_hype:
        rk=adv.where(elig).rank(axis=1,ascending=False)
    else:  # rank among ALL 150 binance coins, then keep HYPE ones (author's definition)
        allQ=P["Q"].where(allC.notna()); allad=allQ.rolling(30,min_periods=15).median()
        alle=allC.notna()&(allC.notna().cumsum()>=60)
        rk=allad.where(alle).rank(axis=1,ascending=False)[syms]
    U=elig&(rk<=top_k)
    Fz=F.where(C.notna()).fillna(0.0)  # listed day with no settlement -> 0
    S=F.rolling(L,min_periods=L).sum() if signal=="F14" else None
    if perm is not None:
        S=pd.DataFrame(S.values[:,perm],index=S.index,columns=S.columns)
    mom=C/C.shift(14)-1
    if venue=="hype":
        PX=P["HC"][syms]; FV=P["HF"][syms]
        U=U&PX.notna()&FV.notna()
    else:
        PX=C; FV=Fz
    H=P["H"][syms].values.reshape(len(C),24,-1) if stop else None
    ix=C.index; t0=ix.get_loc(pd.Timestamp(start,tz="UTC")); t1=len(ix)-1 if end is None else ix.get_loc(pd.Timestamp(end,tz="UTC"))
    c=C.values; px=PX.values; fv=np.nan_to_num(FV.values); s=S.values; mo=mom.values; u=U.values
    m=len(syms); pos=np.zeros(m); entry=np.full(m,np.nan); ban=np.zeros(m,bool)
    POS=[]; RET=[]
    eq=eq0; rows=[]; contrib=np.zeros(m); nst=0; fees=0; fsum=0
    for t in range(t0,t1):
        ts=t-lag  # signal date
        ok=u[ts]&~np.isnan(s[ts])&~np.isnan(px[t])
        if need_next: ok&=~np.isnan(px[t+1])
        ids=np.flatnonzero(ok); N=len(ids); tgt=np.zeros(m)
        if N>=12 or (k_mode=="ter" and N>=6):
            K=N//3 if k_mode=="ter" else int(min(N//3,max(4,np.floor(eq/2/12.5))))
            mr=pd.Series(mo[ts,ids]).rank(pct=True).values
            cs=[ids[i] for i in np.argsort(-s[ts,ids],kind="stable") if (np.isnan(mr[i]) or mr[i]<=momskip)]
            sh=[j for j in cs if not ban[j]][:K]
            lg=[j for j in ids[np.argsort(s[ts,ids],kind="stable")] if j not in sh][:K]
            if sh and lg:
                tgt[sh]=-0.5/len(sh); tgt[lg]=0.5/len(lg)
        ban[:]=False
        tu=tgt*eq
        if k_mode=="cap":
            tu[(np.abs(tu)<10)&(tu!=0)]=0
            d=tu-pos
            tr=(np.abs(d)>=10)|((tu==0)&(pos!=0))|((pos==0)&(tu!=0))
            new=np.where(tr,tu,pos)
        else: new=tu
        fee=np.abs(new-pos).sum()*cost*fee_mult
        newshort=(new<0)&~(pos<0); entry[newshort]=c[t,newshort]; entry[new>=0]=np.nan
        r=px[t+1]/px[t]-1; r=np.where(np.isnan(r),0.0,r)
        hit=np.zeros(m,bool)
        if stop:
            hc=H[t+1]; lvl=entry*(1+stop)
            with np.errstate(invalid="ignore"): cr=(hc>=lvl[None,:])&(new<0)[None,:]
            hit=cr.any(0)
            if hit.any():
                fi=np.argmax(cr,0)
                fill=hc[fi,np.arange(m)]*1.005
                r=r.copy(); r[hit]=fill[hit]/c[t,hit]-1
        pnl_i=new*r
        fund_i=-(new*fv[t+1]*np.where(hit,0.5,1.0)) if fund_on else np.zeros(m)
        sfee=(np.abs(new[hit])*(1+r[hit])).sum()*cost*fee_mult
        contrib+=pnl_i+fund_i-np.abs(new-pos)*cost*fee_mult
        eq+=pnl_i.sum()+fund_i.sum()-fee-sfee; fees+=fee+sfee; fsum+=fund_i.sum(); nst+=hit.sum()
        rows.append((ix[t+1],eq,pnl_i.sum(),fund_i.sum(),fee+sfee,(new>0).sum(),(new<0).sum(),N,new.sum()/max(eq,1e-9)))
        if keep: POS.append(new.copy()); RET.append(r.copy())
        pos=new*(1+r); pos[hit]=0; ban|=hit
        if eq<=0: break
    d=pd.DataFrame(rows,columns=["ts","eq","pnl","fund","fee","nl","ns","N","net"]).set_index("ts")
    d["r"]=d["eq"].pct_change(); d.iloc[0,d.columns.get_loc("r")]=d["eq"].iloc[0]/eq0-1
    x=dict(fees=fees,fund=fsum,stops=int(nst))
    if keep:
        x['pos']=pd.DataFrame(POS,index=d.index,columns=syms); x['ret']=pd.DataFrame(RET,index=d.index,columns=syms)
    return d,pd.Series(contrib,index=syms),x
def stats(d):
    r=d["r"]; eq=d["eq"]
    sh=r.mean()/r.std()*np.sqrt(365); dd=(eq/eq.cummax()-1).min()
    return dict(end=round(eq.iloc[-1],2),sharpe=round(sh,2),maxdd=round(dd,3),cagr=round((eq.iloc[-1]/200)**(365/len(r))-1,3))
