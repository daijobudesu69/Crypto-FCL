"""Descriptive check (no new configs): TOM33 window returns per month-boundary episode, EST vs VAL,
and how much of the EST result the top-5 episodes carry."""
import sys, os, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import famB_calendar as fb
EWb = fb.ew_base(50)
for asset in ["ETH", "EW", "BTC"]:
    x = fb.asset_daily_ret(asset, EWb)
    s = x[fb.WIN["TOM33"]].dropna()
    D = s.index + pd.Timedelta("1D")
    key = (D + pd.Timedelta("5D")).tz_localize(None).to_period("M")
    ep = np.expm1(np.log1p(s).groupby(key).sum())
    ep_est = ep[ep.index < pd.Period("2023-01", "M")]; ep_val = ep[ep.index >= pd.Period("2023-01", "M")]
    srt = ep_est.sort_values(ascending=False); tot = np.log1p(ep_est).sum()
    print(f"{asset}: EST episodes {len(ep_est)} mean {ep_est.mean():+.2%} median {ep_est.median():+.2%} hit {(ep_est>0).mean():.0%} | "
          f"VAL episodes {len(ep_val)} mean {ep_val.mean():+.2%} median {ep_val.median():+.2%} hit {(ep_val>0).mean():.0%}")
    print("   EST top-5 episodes: " + ", ".join(f"{k} {v:+.1%}" for k, v in srt.head(5).items()) +
          f" | EST log-sum {tot:+.2f}, ex top-5 {tot - np.log1p(srt.head(5)).sum():+.2f}")
