"""Step 1: multiple testing. DSR (n=855 desk, 95 hunter, 53 family B, 12 B1-spread signals), Harvey-Liu t>3 and
Bonferroni haircut, 20d circular block bootstrap (10k) CI and P(SR<0.5), circular-shift timing null (state series
rotated against returns)."""
import numpy as np, pandas as pd
from scipy.stats import norm
import dc
lab = dc.lab
d = pd.read_csv(dc.os.path.join(dc.HERE, "disp7_terc_daily.csv"), index_col=0, parse_dates=True)
r = d["r"]; T = len(r); yrs = T / 365
sr = dc.sharpe(r); sk = float(r.skew()); ku = float(r.kurt()) + 3
t_plain = sr * np.sqrt(yrs); t_nw = lab.newey_west_t(r.values, 10)
rl = pd.read_csv(r"C:\Crypto data\backtest data and more\alpha_20261024\hunt\r2_age_dispersion\runlog.csv")
b1 = rl[rl.name.str.match(r"^B1_(breadth50|dbreadth10|corr30|corr90|disp7|disp30)_(MED|TERC)$")]
sd_b1 = float(b1.sharpe.std()); sd_b = float(rl[rl.name.str.startswith("B")].sharpe.std())
lines = ["## Step 1 — multiple testing",
         f"SR {sr:.3f}, T {T} d ({yrs:.2f} y), skew {sk:.2f}, raw kurt {ku:.2f}. t (iid) {t_plain:.2f}, t_NW(10) {t_nw:.2f}.",
         "", "| trials n | DSR (lab default trial-SR sd 0.5) | DSR (trial-SR sd = family-B empirical %.2f) | E[max SR] null (sd 0.5) |" % sd_b, "|---|---|---|---|"]
emc = 0.5772156649
for n in (12, 53, 95, 855):
    a = lab.deflated_sharpe(sr, n, T, sk, ku)
    b = lab.deflated_sharpe(sr, n, T, sk, ku, sr_var_trials=(sd_b / np.sqrt(365)) ** 2)
    e = 0.5 * ((1 - emc) * norm.ppf(1 - 1 / n) + emc * norm.ppf(1 - 1 / (n * np.e)))
    lines.append(f"| {n} | {a:.3f} | {b:.3f} | {e:.2f} |")
lines.append(f"(B1 spread signal family, 12 pre-registered: trial-SR sd {sd_b1:.2f}; family B all 53: sd {sd_b:.2f}.)")
# Harvey-Liu / Bonferroni haircut
p1 = 2 * (1 - norm.cdf(t_nw))
lines += ["", "Harvey-Liu-Zhu: t_NW = %.2f < 3.0 -> FAILS the t>3 hurdle. Bonferroni haircut (two-sided p single = %.4f):" % (t_nw, p1)]
for n in (12, 53, 95, 855):
    pa = min(1.0, p1 * n); ta = norm.ppf(1 - pa / 2) if pa < 1 else 0.0
    lines.append(f"- n={n}: adj p {pa:.3f} -> haircut t {ta:.2f} -> haircut Sharpe {ta / np.sqrt(yrs):.2f} ({(1 - ta / t_nw):.0%} haircut)")
# block bootstrap
rng = np.random.default_rng(20261024)
x = r.values; B = 10000; L = 20; nb = int(np.ceil(T / L))
srs = np.empty(B)
for i in range(B):
    st = rng.integers(0, T, nb)
    idx = (st[:, None] + np.arange(L)[None, :]).ravel()[:T] % T
    y = x[idx]; srs[i] = y.mean() / y.std() * np.sqrt(365)
lo, hi = np.percentile(srs, [5, 95])
lines += ["", f"20-day circular block bootstrap (10,000): median SR {np.median(srs):.2f}, 90% CI [{lo:.2f}, {hi:.2f}], "
          f"P(SR<0.5) = {(srs < 0.5).mean():.3f}, P(SR<0) = {(srs < 0).mean():.3f}, P(SR<1.0) = {(srs < 1.0).mean():.3f}."]
# block bootstrap per year-ish: 60d block
srs60 = np.empty(B); L2 = 60; nb2 = int(np.ceil(T / L2))
for i in range(B):
    st = rng.integers(0, T, nb2)
    idx = (st[:, None] + np.arange(L2)[None, :]).ravel()[:T] % T
    y = x[idx]; srs60[i] = y.mean() / y.std() * np.sqrt(365)
lo2, hi2 = np.percentile(srs60, [5, 95])
lines.append(f"60-day blocks (regime spells avg 27d, so 20d blocks understate dependence): 90% CI [{lo2:.2f}, {hi2:.2f}], P(SR<0.5) = {(srs60 < 0.5).mean():.3f}.")
# circular-shift timing null: rotate the state series vs returns
U = dc.U50(); stt = dc.disp7_state(U)
v = stt.values; ok = np.flatnonzero(np.isfinite(v)); v0 = v[ok]
shifts = list(range(30, len(v0) - 30, 7))
null = []
for k in shifts:
    s2 = pd.Series(np.nan, index=stt.index); s2.iloc[ok] = np.roll(v0, k)
    null.append(dc.sharpe(dc.bt(dc.spread_w(s2, U))["df"]["r"]))
null = np.array(null)
lines.append(f"Circular-shift timing null ({len(null)} rotations of the state path, same spell structure and same time-in-state): "
             f"null SR mean {null.mean():.2f}, sd {null.std():.2f}, 95th pct {np.percentile(null, 95):.2f}, max {null.max():.2f}; "
             f"P(null >= {sr:.2f}) = {(null >= sr).mean():.3f}.")
# always-on spreads (drift check)
alw_l = dc.sharpe(dc.bt(dc.spread_w(pd.Series(1.0, index=stt.index).where(stt.notna()), U))["df"]["r"])
alw_s = dc.sharpe(dc.bt(dc.spread_w(pd.Series(-1.0, index=stt.index).where(stt.notna()), U))["df"]["r"])
lines.append(f"Unconditional spread over the same dates: always LONG-SPREAD SR {alw_l:.2f}, always SHORT-SPREAD SR {alw_s:.2f}.")
dc.append("\n".join(lines))
