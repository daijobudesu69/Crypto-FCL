from common4 import *
C = lab.load("close","1d"); F = lab.load("fund","1d")
CH = lab.load("close","hype_1d"); QH = lab.load("qv","hype_1d"); FH = lab.load("fund","hype_1d")
for s in ["ZECUSDT","XLMUSDT","SANDUSDT","ALGOUSDT","ICPUSDT","DOGEUSDT"]:
    x = pd.DataFrame({"c":CH[s],"qv":QH[s],"f":FH[s],"bc":C[s]}).dropna(subset=["c"])
    print(s, "first", x.index[0].date(), "rows", len(x), "qv>0 first", (x["qv"]>0).idxmax().date() if (x["qv"]>0).any() else None,
          "2024 days qv==0:", int(((x["qv"]==0)&(x.index.year==2024)).sum()), "fund first", FH[s].first_valid_index())
    print(x.loc["2024-03-01":"2024-03-04"])
# market-live definition: qv>0 on the day
live = (QH > 0)
U = lab.universe("1d", top_k=50); UH = lab.hype_universe("1d", top_k=50)
W,_ = candidate_weights(UH, F)
for yr in (2023, 2024):
    m = (C.index.year==yr)&(C.index>=pd.Timestamp("2023-06-01",tz="UTC"))
    tr = (W!=0)[m]
    print(yr, "share of book coin-days where HYPE qv==0 or NaN:", round(float((tr & ~live[m]).sum().sum()/tr.sum().sum()),3))
# base candidate from 2023-07 on full Binance universe
Wf,_ = candidate_weights(U, F)
for st in ["2023-01-01","2023-07-01"]:
    print("base top50 binance from", st, round(lab.backtest(Wf,"1d",start=st)["metrics"]["sharpe"],3), " hype-univ binance", round(lab.backtest(W,"1d",start=st)["metrics"]["sharpe"],3))
# venue universe restricted to live HYPE markets (qv>0 trailing 30d)
livec = live.rolling(30, min_periods=30).sum() >= 25
UV = UH & livec
FB14 = F.rolling(14,min_periods=14).sum(); FH14 = FH.rolling(14,min_periods=14).sum()
print("UV size 2023-07..2024-12 median:", UV.loc["2023-07-01":].sum(axis=1).median())
for tag, sig in [("binF14", -FB14), ("hypeF14", -FH14)]:
    Wv = lab.quantile_ls_weights(sig, UV & sig.notna(), q=0.33)
    a = lab.backtest(Wv,"hype_1d",cost=lab.HYPE_COST,start="2023-07-01")["metrics"]; b = lab.backtest(Wv,"1d",start="2023-07-01")["metrics"]
    print(tag, "HYPE px Sharpe %.2f CAGR %.3f | Binance px Sharpe %.2f | npos %.1f" % (a["sharpe"], a["cagr"], b["sharpe"], a["avg_n_pos"]))
    y = lab.yearly(lab.backtest(Wv,"hype_1d",cost=lab.HYPE_COST,start="2023-07-01")["df"]["r"]); print(y.round(3).T.to_string())
