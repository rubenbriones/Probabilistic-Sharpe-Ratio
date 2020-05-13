import scipy
import numpy as np
import pandas as pd


def estimated_sharpe_ratio(returns):
    """
    Calculate the estimated sharpe ratio (risk_free=0).

    Parameters
    ----------
    returns: list, np.array, pd.Series, pd.DataFrame

    Returns
    -------
    float, pd.Series
    """
    return returns.mean() / returns.std(ddof=1)


def ann_estimated_sharpe_ratio(returns, periods=262):
    """
    Calculate the annualized estimated sharpe ratio (risk_free=0).

    Parameters
    ----------
    returns: list, np.array, pd.Series, pd.DataFrame

    periods: int
        How many items in `returns` complete a Year.
        If returns are daily: 262, weekly: 52, monthly: 12, ...

    Returns
    -------
    float, pd.Series
    """
    sr = estimated_sharpe_ratio(returns)
    sr = sr * np.sqrt(periods)
    return sr


def estimated_sharpe_ratio_std(returns=None, *, skew=None, kurtosis=None, sr=None):
    """
    Calculate the standard deviation of the sharpe ratio estimation.

    Parameters
    ----------
    returns: list, np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass the other 3 parameters.

    skew: float, list, np.array, pd.Series, pd.DataFrame
        The third moment expressed in the same frequency as the other parameters.
        `skew`=0 for normal returns.

    kurtosis: float, list, np.array, pd.Series, pd.DataFrame
        The fourth moment expressed in the same frequency as the other parameters.
        `kurtosis`=3 for normal returns.

    sr: float, list, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.

    Returns
    -------
    float, pd.Series

    Notes
    -----
    This formula generalizes for both normal and non-normal returns.
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """
    if type(returns) != pd.DataFrame:
        _returns = pd.DataFrame(returns)
    else:
        _returns = returns.copy()

    if skew is None:
        skew = pd.Series(scipy.stats.skew(_returns), index=_returns.columns)
    if kurtosis is None:
        kurtosis = pd.Series(scipy.stats.kurtosis(_returns, fisher=False), index=_returns.columns)
    if sr is None:
        sr = estimated_sharpe_ratio(_returns)

    n = len(_returns)
    sr_std = np.sqrt((1 + (0.5 * sr ** 2) - (skew * sr) + (((kurtosis - 3) / 4) * sr ** 2)) / (n - 1))

    if type(returns) != pd.DataFrame:
        sr_std = sr_std.values[0]

    return sr_std


def probabilistic_sharpe_ratio(returns=None, sr_benchmark=0.0, *, sr=None, sr_std=None):
    """
    Calculate the Probabilistic Sharpe Ratio (PSR).

    Parameters
    ----------
    returns: list, np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass a `sr` and `sr_std`.

    sr_benchmark: float
        Benchmark sharpe ratio expressed in the same frequency as the other parameters.
        By default set to zero (comparing against no investment skill).

    sr: float, list, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.

    sr_std: float, list, np.array, pd.Series, pd.DataFrame
        Standard deviation fo the Estimated sharpe ratio,
        expressed in the same frequency as the other parameters.

    Returns
    -------
    float, pd.Series

    Notes
    -----
    PSR = probability that SR^ > SR*
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR* = `sr_benchmark`

    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
        sr_std = estimated_sharpe_ratio_std(returns)

    psr = scipy.stats.norm.cdf((sr - sr_benchmark) / sr_std)

    if type(returns) == pd.DataFrame:
        psr = pd.Series(psr, index=returns.columns)
    elif type(psr) not in (float, np.float64):
        psr = psr[0]

    return psr


def min_track_record_length(returns=None, sr_benchmark=0.0, prob=0.95, *, n=None, sr=None, sr_std=None):
    """
    Calculate the MIn Track Record Length (minTRL).

    Parameters
    ----------
    returns: np.array, pd.Series, pd.DataFrame
        If no `returns` are passed it is mandatory to pass a `sr` and `sr_std`.

    sr_benchmark: float
        Benchmark sharpe ratio expressed in the same frequency as the other parameters.
        By default set to zero (comparing against no investment skill).

    prob: float
        Confidence level used for calculating the minTRL.
        Between 0 and 1, by default=0.95

    n: int
        Number of returns samples used for calculating `sr` and `sr_std`.

    sr: float, np.array, pd.Series, pd.DataFrame
        Sharpe ratio expressed in the same frequency as the other parameters.

    sr_std: float, np.array, pd.Series, pd.DataFrame
        Standard deviation fo the Estimated sharpe ratio,
        expressed in the same frequency as the other parameters.

    Returns
    -------
    float, pd.Series

    Notes
    -----
    minTRL = minimum of returns/samples needed (with same SR and SR_STD) to accomplish a PSR(SR*) > `prob`
    PSR(SR*) = probability that SR^ > SR*
    SR^ = sharpe ratio estimated with `returns`, or `sr`
    SR* = `sr_benchmark`

    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1821643
    """
    if n is None:
        n = len(returns)
    if sr is None:
        sr = estimated_sharpe_ratio(returns)
    if sr_std is None:
        sr_std = estimated_sharpe_ratio_stdev(returns)

    min_trl = 1 + (sr_std ** 2 * (n - 1)) * (scipy.stats.norm.ppf(prob) / (sr - sr_benchmark)) ** 2

    if type(returns) == pd.DataFrame:
        min_trl = pd.Series(min_trl, index=returns.columns)
    elif type(min_trl) not in (float, np.float64):
        min_trl = min_trl[0]

    return min_trl


def skew_to_alpha(skew):
    """
    Convert a skew to alpha parameter needed by scipy.stats.skewnorm(..).

    Parameters
    ----------
    skew: float
        Must be between [-0.999, 0.999] for avoiding complex numbers.

    Returns
    -------
    float
    """
    d = (np.pi / 2 * ((abs(skew) ** (2 / 3)) / (abs(skew) ** (2 / 3) + ((4 - np.pi) / 2) ** (2 / 3)))) ** 0.5
    a = (d / ((1 - d ** 2) ** .5))
    return a * np.sign(skew)


def moments(returns):
    """
    Calculate the four moments: mean, std, skew, kurtosis.

    Parameters
    ----------
    returns: list, np.array, pd.Series, pd.DataFrame

    Returns
    -------
    pd.Series, pd.DataFrame
    """
    if type(returns) != pd.DataFrame:
        return pd.Series({'mean': np.mean(returns),
                          'std': np.std(returns, ddof=1),
                          'skew': scipy.stats.skew(returns),
                          'kurt': scipy.stats.kurtosis(returns, fisher=False)})
    else:
        return returns.apply(moments, axis=1)
