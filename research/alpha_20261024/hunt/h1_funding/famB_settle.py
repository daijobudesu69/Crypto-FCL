"""FAMILY B(ii) - funding-settlement timing on 1h bars (2020-2024 IS).

PRE-REGISTERED EVENT STUDY (written before running):
 events: every (coin, settlement) with coin in the 1d top-50 universe of the PREVIOUS day.
 settlement at time T = close of 1h bar i where fund[i] != 0 (handles 8h and 4h/1h intervals).
 conditioning: realised funding f at T (approx. known ~1h before via predicted rate) AND, as the
 strictly-known robustness check, f_prev = previous settlement's rate for that coin.
 buckets (per settlement): f<=-0.05% | -0.05%<f<-0.005% | baseline 0.005%..0.0125% | 0.0125%..0.05% | 0.05%..0.1% | >=0.1%
 windows (XS-demeaned simple returns vs EW universe, and raw): pre1=[T-1h,T], pre3=[T-3h,T],
          post1=[T,T+1h], post3=[T,T+3h], post8=[T,T+8h]
 t-stats: per-day average then NW t over days (handles clustering).
 RULE GATE: build a trading rule only if a window's gross mean |effect| >= 0.30% (well above 0.14% RT).
"""
from common import *
C = lab.load("close", "1h"); Fh = lab.load("fund", "1h")
U1d = lab.universe("1d", top_k=50)
# map previous-day universe onto hours
Ud = U1d.shift(1).fillna(False).astype(bool)
Uh = Ud.reindex(C.index.floor("1D")).set_axis(C.index)
Uh = Uh.reindex(columns=C.columns).fillna(False).astype(bool).values
R = C.pct_change(fill_method=None).values.astype(np.float64)
lab.clear()
Rm = np.where(Uh, R, np.nan)
ew = np.nanmean(Rm, axis=1)
XR = R - ew[:, None]
Fv = Fh.values.astype(np.float64)
T, N = R.shape
# previous settlement funding per coin (forward fill of nonzero settlements, shifted to exclude current)
Fs = pd.DataFrame(np.where(Fv != 0, Fv, np.nan), index=Fh.index, columns=Fh.columns)
Fprev = Fs.ffill().shift(1).values
del Fs


def csum(A, a, b):
    """sum of A rows i+a .. i+b inclusive for every i (NaN-safe -> NaN if any missing)."""
    out = np.full_like(A, np.nan)
    cs = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(np.nan_to_num(A), axis=0)])
    nn = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(np.isnan(A), axis=0)])
    lo, hi = a, b
    i0 = max(0, -lo); i1 = T - max(0, hi)
    idx = np.arange(i0, i1)
    s = cs[idx + hi + 1] - cs[idx + lo]
    bad = (nn[idx + hi + 1] - nn[idx + lo]) > 0
    s[bad] = np.nan
    out[idx] = s
    return out


win = {"pre1": (0, 0), "pre3": (-2, 0), "post1": (1, 1), "post3": (1, 3), "post8": (1, 8)}
ev = (Fv != 0) & ~np.isnan(Fv) & Uh
ii, jj = np.nonzero(ev)
print("n events", len(ii))
data = {"t": Fh.index[ii], "f": Fv[ii, jj], "fprev": Fprev[ii, jj]}
for k, (a, b) in win.items():
    data["x_" + k] = csum(XR, a, b)[ii, jj]
    data["r_" + k] = csum(R, a, b)[ii, jj]
E = pd.DataFrame(data)
E["hour"] = (E["t"] + pd.Timedelta("1h")).dt.hour
bins = [-1, -0.0005, -0.00005, 0.00005, 0.000125, 0.0005, 0.001, 1]
lbl = ["<=-0.05%", "-0.05..-0.005%", "-0.005..0.005%", "base 0.005..0.0125%", "0.0125..0.05%", "0.05..0.1%", ">=0.1%"]
E["b"] = pd.cut(E["f"], bins, labels=lbl)
E["bp"] = pd.cut(E["fprev"], bins, labels=lbl)
E.to_parquet("famB_settle_events.parquet")


def summarize(col_b):
    rows = []
    for b, g in E.groupby(col_b, observed=True):
        d = {"bucket": b, "n": len(g)}
        for k in win:
            x = g["x_" + k]
            day = x.groupby(g["t"].dt.floor("1D")).mean().dropna()
            d["x_" + k + "_bp"] = x.mean() * 1e4
            d["t_" + k] = lab.newey_west_t(day.values, lags=5)
        rows.append(d)
    return pd.DataFrame(rows)


pd.set_option("display.width", 250)
print("\n== conditioned on realised funding at T (XS-demeaned, bp)")
print(summarize("b").round(2).to_string())
print("\n== conditioned on previous settlement funding (strictly known)")
print(summarize("bp").round(2).to_string())
print("\n== raw (not demeaned) mean bp by realised bucket")
print(E.groupby("b", observed=True)[["r_pre1", "r_pre3", "r_post1", "r_post3", "r_post8"]].mean().mul(1e4).round(2).to_string())
print("\n== by year, bucket >=0.05% (realised), XS-demeaned bp")
hi = E[E["f"] >= 0.0005]
print(hi.groupby(hi["t"].dt.year)[["x_pre3", "x_post1", "x_post3", "x_post8"]].mean().mul(1e4).round(1).assign(n=hi.groupby(hi["t"].dt.year).size()).to_string())
lo = E[E["f"] <= -0.0005]
print("\n== by year, bucket <=-0.05% (realised), XS-demeaned bp")
print(lo.groupby(lo["t"].dt.year)[["x_pre3", "x_post1", "x_post3", "x_post8"]].mean().mul(1e4).round(1).assign(n=lo.groupby(lo["t"].dt.year).size()).to_string())
