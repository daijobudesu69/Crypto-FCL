import sys, time, os, json
T0=time.time()
from indep_sim import *
import numpy as np
which=sys.argv[1]; n=int(sys.argv[2]); budget=float(sys.argv[3])
kw=dict(start="2025-01-01",venue="hype") if which=="fwd" else dict(start="2020-03-01",end="2024-12-31",venue="binance")
m=len([s for s in P["C"].columns if isinstance(MAP.get(s),str) and (which!="fwd" or s in P["HC"].columns)])
fn=f"perm_{which}.json"; out=json.load(open(fn)) if os.path.exists(fn) else []
rng=np.random.default_rng(12345 if which=="fwd" else 999); perms=[rng.permutation(m) for _ in range(n)]
for i in range(len(out),n):
    if time.time()-T0>budget: break
    d,_,_=run(perm=perms[i],**kw); out.append(float(stats(d)["sharpe"])); json.dump(out,open(fn,"w"))
print(which,len(out),"done; mean",round(np.mean(out),2),"max",round(max(out),2))
