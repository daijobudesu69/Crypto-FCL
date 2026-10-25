import is_surv as S, numpy as np, pandas as pd
# instrumented copy: per-coin pnl by side for dead coins, universe C cap
P=S.P; Call=S.Call; rs=list(Call.columns)
src=open("is_surv.py").read()
