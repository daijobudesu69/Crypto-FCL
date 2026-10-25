# Distribusi return & DD 12 bulan dari simulasi audit (review Claude, 25 Okt 2026). Jalankan dengan PYTHONUTF8=1; butuh panel *.pkl di folder audit asli (tidak di repo).
import os, sys, numpy as np, pandas as pd
A=r"C:\Crypto data\backtest data and more\alpha_20261024\audit_2026-10-25"; os.chdir(A)
def load(fn):
    src=open(fn).read().split('if __name__')[0].replace('    return dict(end=','    return d\n    dict(end=')
    g={}; exec(src,g); return g
def stats(nm,d):
    e=d["eq"]; r=e.pct_change().dropna()
    roll=(e.shift(-365)/e-1).dropna()
    dd=[]
    for i in range(0,len(e)-365,7):
        w=e.iloc[i:i+366]; dd.append((w/w.cummax()-1).min())
    dd=np.array(dd)
    q=lambda x,p: round(float(np.percentile(x,p))*100,1)
    print(f"{nm:34s} n12m={len(roll):4d} ret12m p5/25/50/75/95 = {q(roll,5)}/{q(roll,25)}/{q(roll,50)}/{q(roll,75)}/{q(roll,95)}  P(>0)={round((roll>0).mean()*100)}%"
          f" | DD12m p50/p10/worst = {q(dd,50)}/{q(dd,10)}/{round(dd.min()*100,1)}  P(DD>-15%)={round((dd>-0.15).mean()*100)}% | vol {round(r.std()*np.sqrt(365)*100,1)}%", flush=True)
    m=(1+r).groupby(r.index.to_period("M")).prod()-1
    print(f"{'':34s} bulan: median {round(m.median()*100,1)}%  terburuk {round(m.min()*100,1)}%  %positif {round((m>0).mean()*100)}%  | worst day {round(r.min()*100,1)}%")
S=load("surv_sim.py"); allc=set(S["Call"].columns); alive=allc-S["dead"]
for nm,al in [("FWD ALL incl delisted",allc),("FWD survivors",alive),("FWD HYPE-listed today",S["hype"])]:
    stats(nm,S["run"](al,k_mode="cap"))
I=load("is_surv.py"); allc=set(I["Call"].columns); L=set(I["lake"])
for nm,rs,ks in [("IS A author (lake150,HYPE)",L,I["hype"]),("IS C ALL 451 incl dead",allc,allc),("IS D ALL ranked keep HYPE",allc,I["hype"])]:
    stats(nm,I["run"](rs,ks,k_mode="cap"))
