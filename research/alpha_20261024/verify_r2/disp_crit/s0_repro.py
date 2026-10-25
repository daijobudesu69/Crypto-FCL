"""Step 0: reproduce DISP7 TERC headline numbers with an independent implementation."""
import numpy as np, pandas as pd
import dc
lab = dc.lab
U = dc.U50()
st = dc.disp7_state(U)
W = dc.spread_w(st, U)
res = dc.bt(W); m = res["metrics"]; r = res["df"]["r"]
r2 = dc.bt(W, cost=2 * dc.COST)["df"]["r"]; rl = dc.bt(W, extra_lag=1)["df"]["r"]
sw = int((st.dropna().diff().abs() > 0).sum())
sa = dc.align_state(st, r.index)
s1 = r[sa == 1]; s_1 = r[sa == -1]
r.to_frame("r").assign(state=sa).to_csv(dc.os.path.join(dc.HERE, "disp7_terc_daily.csv"))
md = f"""## Step 0 — reproduction
Independent implementation: Sharpe {m['sharpe']:.2f}, CAGR {m['cagr']:+.1%}, MaxDD {m['maxdd']:.1%}, t_NW {m['t_nw']:.2f},
skew {m['skew']:.2f}, excess kurt {m['kurt']:.2f}, days {m['days']} (start {m['start']}), turnover/yr {m['turnover_per_year']:.0f},
cost/yr {m['cost_drag_per_year']:.1%}, funding/yr {m['funding_per_year']:+.1%}, avg n_pos {m['avg_n_pos']:.1f}.
Sub-periods {', '.join(f'{k} {v:.2f}' for k, v in dc.subs(r).items())}; 2x cost {dc.sharpe(r2):.2f}; lag1 {dc.sharpe(rl):.2f}.
Switches {sw}; days in LONG-SPREAD {len(s1)} (ann {s1.mean()*365:+.1%}, SR {dc.sharpe(s1):.2f}), SHORT-SPREAD {len(s_1)} (ann {s_1.mean()*365:+.1%}, SR {dc.sharpe(s_1):.2f}).
Yearly: {dc.yearly_str(r)}.
Claimed: 1.38 / -21% / 1.70, 1.31, 1.11 / 2x 1.28 / lag1 1.43 / ~60 switches -> {'REPRODUCED' if abs(m['sharpe']-1.375)<0.03 else 'MISMATCH'}.
"""
dc.append(md)
