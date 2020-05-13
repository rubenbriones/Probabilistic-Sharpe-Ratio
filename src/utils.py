import scipy.stats as scipy_stats
import numpy as np
import pandas as pd


def skew_to_alpha(skew):
    """
    Convert a skew to alpha parameter needed by scipy_stats.skewnorm(..).

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
    returns: np.array, pd.Series, pd.DataFrame

    Returns
    -------
    pd.Series, pd.DataFrame
    """
    if type(returns) != pd.DataFrame:
        return pd.Series({'mean': np.mean(returns),
                          'std': np.std(returns, ddof=1),
                          'skew': scipy_stats.skew(returns),
                          'kurt': scipy_stats.kurtosis(returns, fisher=False)})
    else:
        return returns.apply(moments, axis=1)
