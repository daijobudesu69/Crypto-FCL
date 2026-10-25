"""Static PNG charts for the report (light mode, reference palette, validated: slots 1-4 pass, contrast WARN -> direct labels)."""
import os, json
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "report_assets")
os.makedirs(OUT, exist_ok=True)
SURF, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948"
GRAY1, GRAY2 = "#52514e", "#a3a29a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.facecolor": SURF, "figure.facecolor": SURF})

EQ = pd.read_csv(os.path.join(HERE, "forward_equity.csv"), index_col=0, parse_dates=True)
R = json.load(open(os.path.join(HERE, "forward_results.json")))


def base(ax, title, sub):
    ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.8); ax.grid(axis="x", visible=False)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9)


# 1. forward equity, $200 start
fig, ax = plt.subplots(figsize=(10, 5.6))
series = [("CHOP30_hype_acct200", "CHOP-30 (HYPE data) — primary", BLUE, 2.6),
          ("CHOP30_binance_acct200", "CHOP-30 (Binance data)", ORANGE, 1.8),
          ("FCL_hype_acct200", "FCL funding (not deployable as-is)", AQUA, 1.8),
          ("DISP7_hype_acct200", "DISP7 dispersion", YELLOW, 1.8),
          ("BENCH_BTC_buy_hold", "BTC buy & hold", GRAY1, 1.4),
          ("BENCH_EW_hype_universe_long", "Equal-weight alts long", GRAY2, 1.4)]
ends = []
for key, lab_, col, lw in series:
    s = EQ[key].dropna()
    s = pd.concat([pd.Series([200.0], index=[s.index[0] - pd.Timedelta("1D")]), s])
    ax.plot(s.index, s.values, color=col, lw=lw, label=lab_, solid_capstyle="round")
    ends.append((s.values[-1], f"${s.values[-1]:.0f}", col))
ends.sort()
ypos = []
for v, txt, col in ends:                         # de-collide end labels
    y = v
    if ypos and y - ypos[-1] < 14:
        y = ypos[-1] + 14
    ypos.append(y)
    ax.annotate(txt, xy=(EQ.index[-1], v), xytext=(8, (y - v) * 0.9), textcoords="offset points",
                color=INK, fontsize=9, va="center")
ax.axhline(200, color=AXIS, lw=1)
ax.set_ylabel("Equity (USD)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
base(ax, r"Forward test 2025-01 → 2026-10, \$200 start, HYPE costs + funding",
     "Specs frozen before the run; run once. Market-neutral books vs. long benchmarks.")
ax.legend(loc="upper left", frameon=False, fontsize=9, labelcolor=INK)
ax.set_xlim(EQ.index[EQ["CHOP30_hype_acct200"].first_valid_index() == EQ.index].min() if False else pd.Timestamp("2024-12-25"), EQ.index[-1] + pd.Timedelta("55D"))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fwd_equity_200usd.png"), dpi=150); plt.close(fig)

# 2. CHOP-30 in-sample $200 account
s = EQ["CHOP30_IS_binance_acct200"].dropna()
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.plot(s.index, s.values, color=BLUE, lw=2)
ax.axhline(200, color=AXIS, lw=1)
for d in ["2020-12-31", "2021-12-31", "2022-12-31", "2023-12-31", "2024-12-30"]:
    v = s.asof(pd.Timestamp(d, tz=s.index.tz) if s.index.tz is not None else pd.Timestamp(d))
    ax.annotate(f"${v:.0f}", xy=(pd.Timestamp(d), v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
ax.set_ylabel("Equity (USD)")
base(ax, r"CHOP-30 backtest 2020-03 → 2024-12 (in-sample), \$200 account, \$10 min order",
     "Binance OHLC as HYPE proxy, HYPE-listed coins, 0.065%/side + funding. Sharpe 1.32, MaxDD −29%.")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "chop_is_equity_200usd.png"), dpi=150); plt.close(fig)

# 3. CHOP-30 forward monthly returns (HYPE data)
m = pd.Series(R["CHOP30_hype_acct200"]["monthly"])
fig, ax = plt.subplots(figsize=(10, 4.2))
cols = [BLUE if v >= 0 else RED for v in m.values]
ax.bar(range(len(m)), m.values * 100, color=cols, width=0.7, edgecolor=SURF, linewidth=2)
ax.axhline(0, color=AXIS, lw=1)
ax.set_xticks(range(len(m))); ax.set_xticklabels([k[2:] for k in m.index], rotation=60, fontsize=8)
for i, v in enumerate(m.values):
    if abs(v) >= 0.04:
        ax.text(i, v * 100 + (0.6 if v > 0 else -0.6), f"{v:+.0%}", ha="center", va="bottom" if v > 0 else "top", fontsize=8, color=INK)
ax.set_ylabel("Monthly return (%)")
base(ax, r"CHOP-30 forward, monthly returns (HYPE data, \$200 account)",
     f"{(m > 0).sum()} of {len(m)} months positive. Blue = gain, red = loss.")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "chop_fwd_monthly.png"), dpi=150); plt.close(fig)
print("charts saved to", OUT)
