"""Signal library (trailing-only; row t uses data up to and including the close of bar t).
Inputs: only open/high/low/close panels (1d) and 4h close (for intraday realized variance)."""
import numpy as np, pandas as pd
from common import lab, BTC, returns


# ---------------------------------------------------------------- family A: asymmetric betas vs BTC
def _rm(X, W, minp):
    return X.rolling(W, min_periods=minp).mean()


def beta_family(W, cond_minp_frac=0.30):
    """Plain, downside (BTC<0), upside (BTC>0) beta and Harvey-Siddique coskewness vs BTC
    over trailing W daily returns."""
    R = returns()
    m = R[BTC]
    M = pd.DataFrame(np.repeat(m.values[:, None], R.shape[1], axis=1), index=R.index, columns=R.columns)
    valid = R.notna() & M.notna()
    Ri = R.where(valid); Mi = M.where(valid)
    minp = int(0.67 * W)

    def beta_on(sel, mp):
        r = Ri.where(sel); x = Mi.where(sel)
        Er, Em = _rm(r, W, mp), _rm(x, W, mp)
        cov = _rm(r * x, W, mp) - Er * Em
        var = _rm(x * x, W, mp) - Em * Em
        return cov / var

    beta = beta_on(valid, minp)
    cmp_ = int(cond_minp_frac * W)
    bminus = beta_on(valid & (M < 0), cmp_)
    bplus = beta_on(valid & (M > 0), cmp_)
    # Harvey-Siddique coskewness: E[e_i * dm^2] / (sqrt(E[e_i^2]) * E[dm^2]), e_i = CAPM residual
    Er, Em = _rm(Ri, W, minp), _rm(Mi, W, minp)
    Em2, Em3 = _rm(Mi ** 2, W, minp), _rm(Mi ** 3, W, minp)
    Erm, Erm2, Er2 = _rm(Ri * Mi, W, minp), _rm(Ri * Mi ** 2, W, minp), _rm(Ri ** 2, W, minp)
    var_m = Em2 - Em ** 2
    var_r = Er2 - Er ** 2
    cov = Erm - Er * Em
    b = cov / var_m
    E_dm3 = Em3 - 3 * Em * Em2 + 2 * Em ** 3
    E_r_dm2 = Erm2 - 2 * Em * Erm + Em ** 2 * Er - Er * var_m
    E_e_dm2 = E_r_dm2 - b * E_dm3
    var_e = (var_r - b ** 2 * var_m).clip(lower=1e-12)
    coskew = E_e_dm2 / (np.sqrt(var_e) * var_m)
    return {"beta": beta, "bminus": bminus, "bplus": bplus, "bdiff": bminus - bplus, "coskew": coskew}


# ---------------------------------------------------------------- family B: volatility structure
def ohlc():
    o = lab.load("open", "1d").astype("float64"); h = lab.load("high", "1d").astype("float64")
    l = lab.load("low", "1d").astype("float64"); c = lab.load("close", "1d").astype("float64")
    return o, h, l, c


def daily_rv_from_4h():
    """Sum of squared 4h log returns inside each UTC day (6 bars) -> daily realized variance.
    Daily index = day open ts (same convention as the 1d panel; known at that day's close)."""
    c4 = lab.load("close", "4h").astype("float64")
    lr = np.log(c4).diff()
    sq = lr ** 2
    day = sq.index.floor("1D")
    rv = sq.groupby(day).sum(min_count=4)          # need >=4 of 6 bars
    lab.clear()                                     # free the 4h panel
    return rv


def skew4h(days=30):
    """Skewness of 4h log returns over trailing `days` days (control factor from round 1), daily index."""
    c4 = lab.load("close", "4h").astype("float64")
    lr = np.log(c4).diff()
    sk = lr.rolling(6 * days, min_periods=int(0.8 * 6 * days)).skew()
    sk = sk[sk.index.hour == 20]                    # value at the last 4h bar of each UTC day
    sk.index = sk.index.floor("1D")
    lab.clear()
    return sk


def vol_structure():
    o, h, l, c = ohlc()
    out = {}
    rv = daily_rv_from_4h().reindex(c.index)
    def rvk(k):
        return np.sqrt(rv.rolling(k, min_periods=max(3, int(0.8 * k))).mean())
    rv60 = rvk(60)
    out["vts5"] = rvk(5) / rv60
    out["vts10"] = rvk(10) / rv60
    pc = c.shift(1)
    tr = (np.maximum(h, pc) - np.minimum(l, pc)) / pc
    tr = tr.where(pc.notna(), (h - l) / c)
    def atr(k):
        return tr.rolling(k, min_periods=max(3, int(0.8 * k))).mean()
    out["atr5_50"] = atr(5) / atr(50)
    out["atr20_100"] = atr(20) / atr(100)
    lrange = np.log((h / l).clip(lower=1.0 + 1e-5))      # Parkinson range (floor avoids log(0))
    llr = np.log(lrange)
    out["vov30"] = llr.rolling(30, min_periods=24).std()
    out["vov90"] = llr.rolling(90, min_periods=72).std()
    pk2 = lrange ** 2 / (4 * np.log(2))
    cc2 = np.log(c / pc) ** 2
    for W in (30, 90):
        mp = int(0.8 * W)
        out[f"pkcc{W}"] = np.sqrt(pk2.rolling(W, min_periods=mp).mean() / cc2.rolling(W, min_periods=mp).mean())
    out["rv60"] = rv60
    return out
