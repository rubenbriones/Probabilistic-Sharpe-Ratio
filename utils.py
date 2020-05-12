import scipy
import numpy as np
import pandas as pd


def sharpe_ratio(returns):
    return returns.mean() / returns.std(ddof=1)


def annualized_sharpe_ratio(returns, periods=262):
    sr = sharpe_ratio(returns)
    sr = sr * np.sqrt(periods)
    return sr


def estimated_sr_std(returns, skew=None, kurtosis=None, sr=None):
    if type(returns) != pd.DataFrame:
        _returns = pd.DataFrame(returns)
    else:
        _returns = returns.copy()

    if skew is None:
        skew = pd.Series(scipy.stats.skew(_returns), index=_returns.columns)
    if kurtosis is None:
        kurtosis = pd.Series(scipy.stats.kurtosis(_returns, fisher=False), index=_returns.columns)
    if sr is None:
        sr = sharpe_ratio(_returns)

    n = len(_returns)
    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))

    if type(returns) != pd.DataFrame:
        sr_std = sr_std.values[0]

    return sr_std


def probabilistic_sharpe_ratio(returns, sr_benchmark=0, sr=None, sr_std=None):
    if sr is None:
        sr = sharpe_ratio(returns)
        sr_std = estimated_sr_std(returns)

    psr = scipy.stats.norm.cdf((sr - sr_benchmark) / sr_std)

    if type(returns) == pd.DataFrame:
        psr = pd.Series(psr, index=returns.columns)
    elif type(psr) not in (float, np.float64):
        psr = psr[0]

    return psr


def skew_to_alpha(s):
    d = (np.pi / 2 * ((abs(s) ** (2 / 3)) / (abs(s) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3)))) ** 0.5
    a = (d / ((1 - d ** 2) ** .5))
    return a * np.sign(s)


def moments(returns):
    if type(returns) != pd.DataFrame:
        return pd.Series({'mean': np.mean(returns),
                          'std': np.std(returns, ddof=1),
                          'skew': scipy.stats.skew(returns),
                          'kurt': scipy.stats.kurtosis(returns, fisher=False)})
    else:
        return returns.apply(moments, axis=1)
