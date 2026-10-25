"""Step 5: risk. Worst day/week/month, worst drawdown episode and its drivers, alt-crash exposure, BTC beta by
state, and the $200 small-account replay (account_sim) of the 11-position version."""
import numpy as np, pandas as pd
import dc
sys_path = r"C:\Crypto data\backtest data and more\alpha_20261024"
import account_sim
C = dc.C; R = dc.R; BTC = dc.BTC
U = dc.U50(); st = dc.disp7_state(U); W = dc.spread_w(st, U)
res = dc.bt(W); d = res["df"]; r = d["r"]
sa = dc.align_state(st, r.index)
btc = R[BTC].reindex(r.index)
alts = U.copy(); alts[BTC] = False
Wa = alts.astype(float).div(alts.sum(1).replace(0, np.nan), axis=0).fillna(0.0)
alt_ew = (Wa.shift(1) * R).sum(axis=1).reindex(r.index)   # EW alt basket held from close t-1
Wl = W.copy(); Wl[BTC] = 0.0; Wbt = W * 0.0; Wbt[BTC] = W[BTC]
r_alt = dc.bt(Wl)["df"]["r"].reindex(r.index).fillna(0.0); r_btc = dc.bt(Wbt)["df"]["r"].reindex(r.index).fillna(0.0)
L = ["## Step 5 — risk"]
wk = (1 + r).resample("W").prod() - 1; mo = (1 + r).resample("ME").prod() - 1
r7 = (1 + r).rolling(7).apply(np.prod, raw=True) - 1; r30 = (1 + r).rolling(30).apply(np.prod, raw=True) - 1
L.append(f"Worst day {r.min():+.1%} ({r.idxmin().date()}, state {sa[r.idxmin()]:+.0f}, BTC {btc[r.idxmin()]:+.1%}, alt EW {alt_ew[r.idxmin()]:+.1%}); "
         f"best day {r.max():+.1%} ({r.idxmax().date()}). Worst calendar week {wk.min():+.1%} (w/e {wk.idxmin().date()}), worst rolling 7d {r7.min():+.1%} (to {r7.idxmin().date()}); "
         f"worst calendar month {mo.min():+.1%} ({mo.idxmin().strftime('%Y-%m')}), worst rolling 30d {r30.min():+.1%} (to {r30.idxmin().date()}). "
         f"Daily vol {r.std() * np.sqrt(365):.0%}/yr; 1st pct day {r.quantile(.01):+.1%}; months negative {(mo < 0).mean():.0%}.")
L.append("Worst 5 days: " + "; ".join(f"{i.date()} {v:+.1%} (st {sa[i]:+.0f}, BTC {btc[i]:+.1%}, altEW {alt_ew[i]:+.1%})" for i, v in r.nsmallest(5).items()))
eq = (1 + r).cumprod(); ddser = eq / eq.cummax() - 1
tr = ddser.idxmin(); pk = eq[:tr].idxmax(); rec = eq[tr:][eq[tr:] >= eq[pk]]
rec_d = rec.index[0].date() if len(rec) else "not recovered by 2024-12-31"
win = (r.index > pk) & (r.index <= tr)
L.append(f"Worst drawdown {ddser.min():.1%}: peak {pk.date()} -> trough {tr.date()} ({win.sum()} d), recovered {rec_d}. "
         f"During it: alt leg {r_alt[win].sum():+.1%}, BTC leg {r_btc[win].sum():+.1%} (arith.), days in LONG-SPREAD {(sa[win] == 1).sum()}, SHORT-SPREAD {(sa[win] == -1).sum()}, "
         f"switches {int((sa[win].diff().abs() > 0).sum())}; BTC {(1 + btc[win]).prod() - 1:+.1%}, alt EW {(1 + alt_ew[win]).prod() - 1:+.1%}.")
# other drawdowns > 10%
dd_ep = []
inn = False
for i, v in ddser.items():
    if v < 0 and not inn:
        inn = True; s0 = i
    if v == 0 and inn:
        inn = False; seg = ddser[s0:i]; dd_ep.append((s0, seg.idxmin(), i, seg.min()))
if inn:
    seg = ddser[s0:]; dd_ep.append((s0, seg.idxmin(), None, seg.min()))
big = sorted([e for e in dd_ep if e[3] < -0.10], key=lambda e: e[3])
L.append("All drawdowns deeper than -10%: " + "; ".join(f"{a.date()}->{b.date()} {v:.1%} (rec {c.date() if c is not None else 'open'})" for a, b, c, v in big))
# end-2024 drawdown
L.append(f"Drawdown at 2024-12-31: {ddser.iloc[-1]:.1%}; equity high set on {eq.idxmax().date()}.")
# crash windows
ev = [("2021-01-21 dip", "2021-01-20", "2021-01-23"), ("2021-05-12..23 China/Elon crash", "2021-05-12", "2021-05-24"),
      ("2021-05-19 alone", "2021-05-19", "2021-05-20"), ("2021-12-04 flash crash", "2021-12-03", "2021-12-06"),
      ("2022-05-05..13 LUNA", "2022-05-05", "2022-05-14"), ("2022-06-10..19 3AC/Celsius", "2022-06-10", "2022-06-20"),
      ("2022-11-06..12 FTX", "2022-11-06", "2022-11-13"), ("2023-03-09..14 SVB", "2023-03-09", "2023-03-15"),
      ("2023-06-05..12 SEC suits", "2023-06-05", "2023-06-13"), ("2024-03-11..20 alt top", "2024-03-11", "2024-03-21"),
      ("2024-04-12..14 Iran", "2024-04-12", "2024-04-15"), ("2024-08-02..06 yen carry", "2024-08-02", "2024-08-07"),
      ("2024-12-09..10 alt flush", "2024-12-09", "2024-12-11")]
L += ["", "| event (return dates) | state(s) | strategy | BTC | alt EW | alt leg | BTC leg |", "|---|---|---|---|---|---|---|"]
for nm, a, b in ev:
    m = (r.index >= pd.Timestamp(a, tz="UTC")) & (r.index < pd.Timestamp(b, tz="UTC"))
    if m.sum() == 0:
        continue
    sts = "/".join(f"{x:+.0f}" for x in sorted(set(sa[m].dropna())))
    L.append(f"| {nm} | {sts} | {(1 + r[m]).prod() - 1:+.1%} | {(1 + btc[m]).prod() - 1:+.1%} | {(1 + alt_ew[m]).prod() - 1:+.1%} | {r_alt[m].sum():+.1%} | {r_btc[m].sum():+.1%} |")
# beta by state
L.append("")
for sv in (1, -1):
    m = (sa == sv) & btc.notna()
    b = np.polyfit(btc[m], r[m], 1)[0]
    ba = np.polyfit(alt_ew[m], r[m], 1)[0]
    dn = m & (btc < -0.03); up = m & (btc > 0.03)
    L.append(f"State {sv:+d}: BTC beta {b:+.2f} (corr {np.corrcoef(btc[m], r[m])[0, 1]:+.2f}), beta to alt EW {ba:+.2f}; "
             f"on BTC < -3% days (n={dn.sum()}) mean strat {r[dn].mean():+.2%}, on BTC > +3% days (n={up.sum()}) {r[up].mean():+.2%}; "
             f"on alt-EW < -5% days (n={(m & (alt_ew < -0.05)).sum()}) mean strat {r[m & (alt_ew < -0.05)].mean():+.2%}.")
ab = (pd.concat([alt_ew, btc], axis=1).dropna())
L.append(f"Trailing reality check: full-sample beta of alt EW to BTC {np.polyfit(ab.iloc[:, 1], ab.iloc[:, 0], 1)[0]:.2f} -> the 0.5/0.5 spread is NOT beta-neutral: "
         f"LONG-SPREAD carries net long beta, SHORT-SPREAD net short beta. Full-sample corr strat vs BTC {r.corr(btc):+.2f}.")
dc.append("\n".join(L))
# small-account replay
U11 = dc.U50(11)
W11 = dc.spread_w(st, U11)
L2 = ["Small-account replay (account_sim, $10 min order, HYPE cost, signal from U50, traded on HYPE top-11 = 10 alts + BTC):", "",
      "| start equity | end equity | total ret | MaxDD | orders | skipped < $10 | fees $ | funding $ | avg n_pos |", "|---|---|---|---|---|---|---|---|---|"]
for e0 in (200.0, 300.0):
    out, sm = account_sim.simulate(W11, "1d", equity0=e0, min_order=10.0, cost=dc.COST, start="2020-06-29")
    L2.append(f"| ${e0:.0f} | ${sm['equity_end']:.0f} | {sm['total_return']:+.0%} | {sm['maxdd']:.1%} | {sm['orders']} | {sm['skipped_below_min']} | {sm['fees_usd']:.0f} | {sm['funding_usd']:+.0f} | {sm['avg_n_pos']:.1f} |")
    rr = out["equity"].pct_change().dropna()
    L2.append(f"|  (SR {dc.sharpe(rr):.2f}; 2024 {(out['equity'][out.index.year == 2024].iloc[-1] / out['equity'][out.index.year == 2023].iloc[-1] - 1):+.0%}; 2023-24 SR {dc.sharpe(dc.between(rr, '2023-01-01', '2025-01-01')):.2f}) | | | | | | | | |")
r11 = dc.bt(W11)["df"]["r"]
L2.append(f"lab.backtest of the same 11-pos weights: SR {dc.sharpe(r11):.2f}, subs {', '.join(f'{k} {v:.2f}' for k, v in dc.subs(r11).items())}; {dc.yearly_str(r11)}.")
dc.append("\n".join(L2))
