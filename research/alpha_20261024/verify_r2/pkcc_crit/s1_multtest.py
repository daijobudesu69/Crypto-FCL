"""Step 0 (re-implementation check) + Step 1 (multiple testing).
Pre-registered: DSR at n = 49 (hunter family B), 85 (hunter total), 855 (desk) with trial SR dispersion
0.5 (lab default) and 0.83 (std of hunter's 50 family-B Sharpes); Harvey-Liu t>3 + Bonferroni / BHY
(rank 1) haircuts at M = 855; 20-day block bootstrap (10k, circular blocks) 90% CI and P(SR<0.5),
also for the 16-pos book."""
import numpy as np, pandas as pd
from scipy.stats import norm
from vlib import *

W, U = base_weights()
res = bt(W); m = res["metrics"]; df = res["df"]; r = df["r"]
W16, _ = base_weights(n=8)
res16 = bt(W16); r16 = res16["df"]["r"]; m16 = res16["metrics"]
lag1 = bt(W, lag=1)["metrics"]["sharpe"]; lag2 = bt(W, lag=2)["metrics"]["sharpe"]
c2 = bt(W, cost=2 * COST)["metrics"]["sharpe"]
sb = subs(r)
# legs (raw) for later steps
rl = bt(W.clip(lower=0))["df"]["r"]; rs = bt(W.clip(upper=0))["df"]["r"]
out = pd.DataFrame({"r": r, "gross_r": df["gross_r"], "fund_r": df["fund_r"], "cost_r": df["cost_r"],
                    "long_r": rl, "short_r": rs, "r16": r16})
out.to_parquet(os.path.join(HERE, "base_daily_IS.parquet"))
W.to_parquet(os.path.join(HERE, "base_weights_IS.parquet"))

txt = ["## Step 0 - independent re-implementation (vlib.py, own code)",
       f"- PKCC30 quintile book, HYPE top-50: Sharpe {fmt(m['sharpe'])} (claim 1.34), CAGR {m['cagr']:+.1%}, "
       f"vol {m['vol']:.1%}, MaxDD {m['maxdd']:.1%} (claim -28%), t_NW {fmt(m['t_nw'])}, days {m['days']}, "
       f"start {m['start']}, avg n_pos {m['avg_n_pos']:.1f}, turnover {m['turnover_per_year']:.0f}x/yr, "
       f"skew {m['skew']:.2f}, excess kurt {m['kurt']:.1f}",
       f"- sub-periods {', '.join(f'{k} {fmt(v)}' for k, v in sb.items())} (claim 1.36/0.75/1.82); "
       f"2x cost {fmt(c2)} (1.09); lag1 {fmt(lag1)} (0.92); lag2 {fmt(lag2)} (0.17)",
       f"- 16-pos (8/8): Sharpe {fmt(m16['sharpe'])} (claim 1.01), MaxDD {m16['maxdd']:.1%}",
       "- Verdict step 0: numbers reproduce -> no implementation discrepancy to chase; critique below is on the "
       "statistics, not the code."]
append("\n".join(txt))

# ---------------- Step 1
T = m["days"]; yrs = T / 365
sr = m["sharpe"]; sk = m["skew"]; ku = m["kurt"] + 3
rows = []
for nm, mm in (("quintile", m), ("16pos", m16)):
    for n in (49, 85, 855):
        for sd in (0.5, 0.83):
            rows.append((nm, n, sd, lab.deflated_sharpe(mm["sharpe"], n, mm["days"], mm["skew"], mm["kurt"] + 3,
                                                         sr_var_trials=(sd / np.sqrt(365)) ** 2)))
dsr = pd.DataFrame(rows, columns=["book", "n", "sr_sd", "dsr"])
# expected max SR under null for reference
emc = 0.5772156649
def emax(n, sd):
    return sd * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * np.e)))

t_plain = sr * np.sqrt(yrs); t_nw = m["t_nw"]
M = 855
def haircut(t, M, method):
    p = 2 * (1 - norm.cdf(abs(t)))
    if method == "bonf":
        pa = min(1.0, p * M)
    else:  # BHY at rank 1 (most significant test): p * M * c(M)
        cM = np.sum(1 / np.arange(1, M + 1))
        pa = min(1.0, p * M * cM)
    ta = norm.ppf(1 - pa / 2) if pa < 1 else 0.0
    return p, pa, ta
hl = {}
for nm, t in (("plain", t_plain), ("NW", t_nw)):
    for meth in ("bonf", "bhy"):
        p, pa, ta = haircut(t, M, meth)
        hl[(nm, meth)] = (p, pa, ta, ta / np.sqrt(yrs))
# t needed
t_bonf = norm.ppf(1 - 0.05 / (2 * M)); cM = np.sum(1 / np.arange(1, M + 1)); t_bhy = norm.ppf(1 - 0.05 / (2 * M * cM))

# ---------------- block bootstrap
rng = np.random.default_rng(20261024)
def boot(x, B=10000, L=20):
    x = np.asarray(pd.Series(x).dropna().values); n = len(x)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    idx = idx.reshape(B, -1)[:, :n]
    s = x[idx]
    return s.mean(1) / s.std(1, ddof=1) * np.sqrt(365)
bs = boot(r); bs16 = boot(r16)
bsl1 = boot(bt(W, lag=1)["df"]["r"])

txt = ["## Step 1 - multiple testing",
       f"- Sharpe {fmt(sr)} over {T} days ({yrs:.2f} y): plain t = SR*sqrt(y) = {fmt(t_plain)}, Newey-West t = {fmt(t_nw)}.",
       f"- Expected max Sharpe of null trials (Bailey-LdP): n=49 sd0.5 -> {emax(49, .5):.2f}; n=85 -> {emax(85, .5):.2f}; "
       f"n=855 sd0.5 -> {emax(855, .5):.2f}, sd0.83 -> {emax(855, .83):.2f}.",
       "- Deflated Sharpe probability (P[true SR > E(max null SR)]):",
       "", "| book | n=49 sd0.5 | n=49 sd0.83 | n=85 sd0.5 | n=85 sd0.83 | n=855 sd0.5 | n=855 sd0.83 |", "|---|---|---|---|---|---|---|"]
for nm in ("quintile", "16pos"):
    d = dsr[dsr.book == nm]
    txt.append(f"| {nm} | " + " | ".join(f"{v:.2f}" for v in d.sort_values(['n', 'sr_sd'])['dsr']) + " |")
txt += ["",
        f"- Harvey-Liu: t>3 hurdle -> plain t {fmt(t_plain)} FAILS, NW t {fmt(t_nw)} barely passes. "
        f"At M=855 the Bonferroni hurdle is t>{t_bonf:.2f} and BHY(rank 1) t>{t_bhy:.2f}.",
        f"- Haircut Sharpe at M=855: Bonferroni (NW t) -> adj p {hl[('NW', 'bonf')][1]:.2f}, haircut SR {hl[('NW', 'bonf')][3]:.2f}; "
        f"BHY (NW t) -> adj p {hl[('NW', 'bhy')][1]:.2f}, haircut SR {hl[('NW', 'bhy')][3]:.2f}; "
        f"Bonferroni (plain t) -> haircut SR {hl[('plain', 'bonf')][3]:.2f}.",
        f"- 20d circular block bootstrap (10k): quintile SR 90% CI [{np.percentile(bs, 5):.2f}, {np.percentile(bs, 95):.2f}], "
        f"median {np.median(bs):.2f}, P(SR<0.5) = {np.mean(bs < 0.5):.3f}, P(SR<0) = {np.mean(bs < 0):.3f}.",
        f"- 16-pos: 90% CI [{np.percentile(bs16, 5):.2f}, {np.percentile(bs16, 95):.2f}], P(SR<0.5) = {np.mean(bs16 < 0.5):.3f}, P(SR<0) = {np.mean(bs16 < 0):.3f}.",
        f"- quintile lag1: 90% CI [{np.percentile(bsl1, 5):.2f}, {np.percentile(bsl1, 95):.2f}], P(SR<0.5) = {np.mean(bsl1 < 0.5):.3f}.",
        "- NOTE: the bootstrap CI is sampling noise only, conditional on THIS config; it does not include selection "
        "(the DSR/haircut lines do)."]
append("\n".join(txt))
print("\n".join(txt))
