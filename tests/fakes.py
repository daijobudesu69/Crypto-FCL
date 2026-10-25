import numpy as np
import pandas as pd


def candles(n_days=80, end="2026-10-24", drift=0.0, vol=1e6, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(end=pd.Timestamp(end, tz="UTC"), periods=n_days, freq="1D")
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.02, n_days)))
    return pd.DataFrame({"ts": ts, "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
                         "volume": np.full(n_days, vol / 100)})


class FakeInfo:
    """pf=Exception -> predicted_fundings() raise; fail = koin yang candle-nya raise."""

    def __init__(self, coins, mids=None, pf=None, fh=None, fail=()):
        self.coins = coins
        self.mids = mids or {c: 100.0 for c in coins}
        self.pf = pf or {}
        self.fh = fh or {}
        self.fail = set(fail)

    def predicted_fundings(self):
        if isinstance(self.pf, Exception):
            raise self.pf
        return self.pf

    def all_mids(self):
        return dict(self.mids)

    def candles(self, coin, interval, start_ms):
        if coin in self.fail:
            raise RuntimeError("HYPE /info candleSnapshot gagal")
        i = self.coins.index(coin)
        return candles(seed=i, vol=1e6 * (len(self.coins) - i))

    def funding_history(self, coin, start_ms, end_ms):
        return [x for x in self.fh.get(coin, []) if start_ms <= x[0] <= end_ms]
