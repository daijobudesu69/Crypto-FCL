"""IS-only reproduction of the frozen CHOP-30 / DISP7-200 specs before the forward test. No ALPHA_FORWARD."""
import os, sys, json
assert os.environ.get("ALPHA_FORWARD") != "1"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import strategies as S, lab
P, U = S.venue_panels("binance")
out = {}
for name, mk in [("CHOP30", S.chop_targets), ("DISP7", S.disp_targets)]:
    tg, rt = mk(P, U)
    for lbl, kw in [("ideal_2020", dict(start="2020-03-01", ideal=True)), ("acct200_2020", dict(start="2020-03-01")),
                    ("acct200_2023", dict(start="2023-01-01"))]:
        d, s = S.replay(P, tg, rt, end="2024-12-31", **kw)
        y = lab.yearly(d["r"])["ret"].round(3).to_dict()
        out[f"{name}_{lbl}"] = {**s, "yearly": {int(k): v for k, v in y.items()}}
        print(f"{name:7s} {lbl:13s} ${s['equity_end']:7.0f} SR {s['sharpe']:.2f} CAGR {s['cagr']:+.1%} DD {s['maxdd']:.1%} npos {s['avg_n_pos']:.1f} fees ${s['fees']:.0f} fund ${s['funding']:+.0f} orders {s['orders']} yrs {y}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "is_check_r2.json"), "w"), indent=1, default=str)
