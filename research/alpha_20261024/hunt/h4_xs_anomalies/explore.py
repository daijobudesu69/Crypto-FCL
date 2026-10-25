"""Initial data exploration (panel shape, universe size, funding coverage, controls)."""
import sys; sys.path.insert(0, r"C:\Crypto data\backtest data and more\alpha_20261024")
import lab, numpy as np, pandas as pd
C = lab.load("close","1d")
print(C.shape, C.index[0], C.index[-1])
print(C.notna().sum(axis=1).resample("QE").last())
U = lab.universe("1d", top_k=50)
print("U50 size", U.sum(axis=1).resample("QE").last().values)
F = lab.load("fund","1d")
print("fund coverage", F.notna().sum(axis=1).resample("QE").last().values)
ctrl = lab.controls(); print(ctrl.describe())
