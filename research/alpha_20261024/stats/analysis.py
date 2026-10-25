"""Points 2-4 of the user's follow-up: RNT comparison, fat tails vs bell curve, 4 hypothesis tests.
Uses already-frozen strategies and already-run forward results (no re-selection)."""
import os, sys, json
os.environ["ALPHA_FORWARD"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "final"))
import numpy as np, pandas as pd
from scipy import stats as st
import lab, fcl, strategies as S
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUTD = os.path.join(ROOT, "report_assets"); FWD = pd.Timestamp("2025-01-01", tz="UTC")
SURF, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
COL = {"FCL": "#1baf7a", "CHOP-30": "#2a78d6", "RNT": "#eb6834", "BTC": "#52514e"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})
RES = {}

# ------------------------------------------------------------------ daily return series ($200 accounts)
eqf = pd.read_csv(os.path.join(ROOT, "final", "forward_equity.csv"), index_col=0, parse_dates=True)
def rets(eq, start_eq=200.0):
    eq = eq.dropna(); r = eq.pct_change(); r.iloc[0] = eq.iloc[0] / start_eq - 1; return r
d_is, s_is, _ = fcl.run("2020-03-01", "2024-12-31", venue="binance")
R = {"FCL": pd.concat([d_is["r"], rets(eqf["FCL_hype_acct200"])]),
     "CHOP-30": pd.concat([rets(eqf["CHOP30_IS_binance_acct200"]), rets(eqf["CHOP30_hype_acct200"])])}
rnt_dir = r"C:\Crypto data\Crypto-RS-Neutral & Trend\research\audit_response"
ri = pd.read_csv(os.path.join(rnt_dir, "v11_is_daily.csv"), parse_dates=["date"], index_col="date")["equity"]
ro = pd.read_csv(os.path.join(rnt_dir, "v11_oos_daily.csv"), parse_dates=["date"], index_col="date")["equity"]
ri = ri[ri.ne(200.0).cummax()]                                    # drop the flat warm-up
R["RNT"] = pd.concat([rets(ri), rets(ro)])
btc = lab.load("close", "1d")["BTCUSDT"]; R["BTC"] = btc.pct_change().loc["2020-03-01":]
for k in R:
    R[k].index = pd.DatetimeIndex(R[k].index).tz_localize(None) if pd.DatetimeIndex(R[k].index).tz is not None else pd.DatetimeIndex(R[k].index)
    R[k] = R[k][~R[k].index.duplicated()].dropna()
FW = pd.Timestamp("2025-01-01")
def per(k, p):
    r = R[k]
    return r[r.index < FW] if p == "IS" else (r[r.index >= FW] if p == "FWD" else r)

def summ(r):
    m = lab.metrics(r, "1d"); eq = 200 * (1 + r).cumprod()
    return {"start": str(r.index[0].date()), "end": str(r.index[-1].date()), "end_usd_from_200": float(eq.iloc[-1]),
            "cagr": m["cagr"], "vol": m["vol"], "sharpe": m["sharpe"], "sortino": m["sortino"], "maxdd": m["maxdd"],
            "worst_month": m["worst_month"], "pct_pos_months": m["pct_pos_months"], "skew": m["skew"], "kurt_excess": m["kurt"]}

# ------------------------------------------------------------------ 2. comparison with RNT
cmp = {p: {k: summ(per(k, p)) for k in R} for p in ["IS", "FWD"]}
corr = {p: pd.DataFrame({k: per(k, p) for k in R}).dropna().corr().round(2).to_dict() for p in ["IS", "FWD"]}
combo = {}
for p in ["IS", "FWD"]:
    df = pd.DataFrame({k: per(k, p) for k in R}).dropna()
    for a, b in [("FCL", "RNT"), ("CHOP-30", "RNT"), ("FCL", "CHOP-30")]:
        combo[f"{p}: 50% {a} + 50% {b}"] = summ(0.5 * df[a] + 0.5 * df[b])
RES["comparison"] = cmp; RES["correlation"] = corr; RES["combos"] = combo

# yearly table
yr = {k: {int(y): round(float(v), 4) for y, v in lab.yearly(R[k])["ret"].items()} for k in R}
RES["yearly"] = yr

def style(ax, title, sub=None):
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.7)
    for s in ["top", "right"]: ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=10.5, fontweight="bold", pad=16 if sub else 6)
    if sub: ax.text(0, 1.0, sub, transform=ax.transAxes, color=INK2, fontsize=8, va="bottom")

# chart: equity forward FCL / CHOP / RNT / BTC
fig, ax = plt.subplots(figsize=(10, 4.8))
for k in R:
    r = per(k, "FWD"); eq = 200 * (1 + r).cumprod()
    ax.plot(eq.index, eq.values, color=COL[k], lw=2.4 if k != "BTC" else 1.4, label=k)
    ax.annotate(f"{k} ${eq.iloc[-1]:.0f}", xy=(eq.index[-1], eq.iloc[-1]), xytext=(6, 0), textcoords="offset points", fontsize=8.5, color=INK, va="center")
ax.axhline(200, color=AXIS, lw=1); ax.set_ylabel("Equity (USD)"); ax.set_xlim(None, eq.index[-1] + pd.Timedelta("80D"))
style(ax, "Forward 2025-01 → 2026-09/10: FCL vs CHOP-30 vs RNT v1.1 vs BTC, $200 start")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "cmp_fwd_equity_rnt.png"), dpi=150); plt.close(fig)

# ------------------------------------------------------------------ 3. fat tails vs bell curve (full period)
fat = {}
fig, axs = plt.subplots(2, 4, figsize=(15, 7.2), gridspec_kw={"height_ratios": [1.15, 1]})
for i, k in enumerate(R):
    r = per(k, "ALL"); z = (r - r.mean()) / r.std()
    jb = st.jarque_bera(r)
    n = len(r); exp3 = n * 2 * st.norm.sf(3); exp4 = n * 2 * st.norm.sf(4)
    fat[k] = {"days": n, "skew": float(st.skew(r)), "kurt_excess": float(st.kurtosis(r)),
              "jarque_bera_p": float(jb.pvalue), "days_beyond_3sd": int((z.abs() > 3).sum()), "expected_if_normal_3sd": round(exp3, 1),
              "days_beyond_4sd": int((z.abs() > 4).sum()), "expected_if_normal_4sd": round(exp4, 2),
              "worst_day": float(r.min()), "worst_day_sd": float(z.min()), "best_day": float(r.max()), "best_day_sd": float(z.max()),
              "VaR99_hist": float(-r.quantile(0.01)), "VaR99_normal": float(-(r.mean() + st.norm.ppf(0.01) * r.std())),
              "CVaR99_hist": float(-r[r <= r.quantile(0.01)].mean())}
    ax = axs[0, i]
    bins = np.linspace(-6, 6, 61)
    ax.hist(z.clip(-6, 6), bins=bins, density=True, color=COL[k], alpha=0.85, edgecolor=SURF, linewidth=0.6)
    xx = np.linspace(-6, 6, 400); ax.plot(xx, st.norm.pdf(xx), color=INK, lw=1.4, ls="--", label="normal (bell curve)")
    ax.set_yscale("log"); ax.set_ylim(1e-4, 1.2); ax.set_xlabel("daily return, in standard deviations")
    style(ax, f"{k}", f"kurtosis +{fat[k]['kurt_excess']:.1f}, skew {fat[k]['skew']:+.2f}, >3σ days {fat[k]['days_beyond_3sd']} vs {exp3:.0f} normal")
    if i == 0: ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax2 = axs[1, i]
    (osm, osr), (sl, ic, _) = st.probplot(z, dist="norm")
    ax2.scatter(osm, osr, s=6, color=COL[k]); lim = max(abs(osr).max(), 4.5)
    ax2.plot([-lim, lim], [-lim, lim], color=INK, lw=1, ls="--")
    ax2.set_xlabel("normal quantile"); ax2.set_ylabel("actual quantile (σ)")
    style(ax2, f"{k}: QQ plot")
fig.suptitle("Fat tails vs bell curve: daily returns 2020-03 → 2026-10 (log-scale histograms)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "fat_tails_bell_curve.png"), dpi=140); plt.close(fig)
RES["fat_tails"] = fat

# ------------------------------------------------------------------ 4. hypothesis tests
def boot_sharpe(r, reps=5000, block=20, seed=1):
    rng = np.random.default_rng(seed); x = r.values; n = len(x); out = np.empty(reps)
    for b in range(reps):
        idx = []; i = rng.integers(n)
        while len(idx) < n:
            idx.append(i); i = (i + 1) % n if rng.random() > 1 / block else rng.integers(n)
        y = x[np.array(idx[:n])]; out[b] = y.mean() / y.std() * np.sqrt(365)
    return out

tests = {}; boots = {}
for p in ["IS", "FWD"]:
    for k in R:
        r = per(k, p); sr = r.mean() / r.std() * np.sqrt(365)
        t = lab.newey_west_t(r.values, 10); pval = float(st.norm.sf(t))
        b = boot_sharpe(r); boots[(p, k)] = b
        T = len(r)
        dsr = {n: lab.deflated_sharpe(sr, n, T, float(st.skew(r)), float(st.kurtosis(r)) + 3) for n in [1, 10, 100, 900]}
        tests[f"{p}|{k}"] = {"sharpe": float(sr), "t_nw": float(t), "p_one_sided": pval, "boot_ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))],
                             "boot_p_sr_le_0": float((b <= 0).mean()), "dsr": dsr, "days": T}
RES["tests"] = tests

# ---- 4 charts
models = list(R)
# (1) t-test
fig, ax = plt.subplots(figsize=(9, 4.3)); x = np.arange(len(models)); w = 0.36
for j, (p, a) in enumerate([("IS", 0.55), ("FWD", 1.0)]):
    vals = [tests[f"{p}|{k}"]["t_nw"] for k in models]
    bars = ax.bar(x + (j - 0.5) * w, vals, w * 0.92, color=[COL[k] for k in models], alpha=a, edgecolor=SURF, linewidth=2)
    for xi, v in zip(x + (j - 0.5) * w, vals): ax.text(xi, v + (0.08 if v >= 0 else -0.08), f"{p}\n{v:.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
for yv, lb in [(1.65, "p=0.05 (t 1.65)"), (3.0, "Harvey-Liu bar (t 3)")]:
    ax.axhline(yv, color=INK2, lw=1, ls="--"); ax.text(len(models) - 0.45, yv, lb, fontsize=8, color=INK2, va="bottom", ha="right")
ax.axhline(0, color=AXIS, lw=1); ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("Newey-West t-stat of mean daily return")
style(ax, "Test 1: t-test, H0 = mean daily return is zero", "light bar = 2020-24 in-sample, solid = 2025-26 forward")
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "test1_ttest.png"), dpi=150); plt.close(fig)
# (2) bootstrap
fig, axs = plt.subplots(1, 4, figsize=(15, 3.6), sharey=False)
for ax, k in zip(axs, models):
    b = boots[("FWD", k)]; ax.hist(b, bins=60, color=COL[k], alpha=0.85, edgecolor=SURF, linewidth=0.4)
    ax.axvline(0, color=INK, lw=1.2); ax.axvline(tests[f"FWD|{k}"]["sharpe"], color=INK, lw=1.2, ls="--")
    lo, hi = tests[f"FWD|{k}"]["boot_ci90"]
    style(ax, k, f"forward Sharpe {tests[f'FWD|{k}']['sharpe']:.2f}, 90% CI [{lo:.2f}, {hi:.2f}], P(≤0) {tests[f'FWD|{k}']['boot_p_sr_le_0']:.0%}")
    ax.set_xlabel("bootstrapped annual Sharpe")
fig.suptitle("Test 2: block bootstrap (20-day blocks, 5000 resamples) of the forward Sharpe", x=0.01, ha="left", fontsize=11.5, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "test2_bootstrap.png"), dpi=140); plt.close(fig)
# (4) deflated Sharpe vs number of trials (full period)
fig, ax = plt.subplots(figsize=(9, 4.3)); ns = np.unique(np.logspace(0, 3.2, 30).astype(int))
for k in models:
    r = per(k, "ALL"); sr = r.mean() / r.std() * np.sqrt(365)
    ys = [lab.deflated_sharpe(sr, n, len(r), float(st.skew(r)), float(st.kurtosis(r)) + 3) for n in ns]
    ax.plot(ns, ys, color=COL[k], lw=2.2, label=f"{k} (Sharpe {sr:.2f})")
    RES.setdefault("dsr_full", {})[k] = {int(n): float(y) for n, y in zip(ns, ys)}
ax.axhline(0.95, color=INK2, lw=1, ls="--"); ax.text(1.1, 0.955, "95% confidence", fontsize=8, color=INK2)
ax.axvline(900, color=INK2, lw=1, ls=":"); ax.text(880, 0.05, "~900 configs tried by this desk", fontsize=8, color=INK2, ha="right")
ax.set_xscale("log"); ax.set_ylim(0, 1.02); ax.set_xlabel("number of strategies tried (log)"); ax.set_ylabel("P(true Sharpe > luck of the best trial)")
style(ax, "Test 4: deflated Sharpe ratio (multiple testing), full period 2020-03 → 2026-10")
ax.legend(frameon=False, fontsize=8.5, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(OUTD, "test4_deflated_sharpe.png"), dpi=150); plt.close(fig)

json.dump(RES, open(os.path.join(HERE, "analysis_results.json"), "w"), indent=1, default=str)
print("done")
