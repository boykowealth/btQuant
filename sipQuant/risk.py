"""
sipQuant.risk — Risk metrics.
Pure NumPy. No pandas, scipy, or external dependencies.

Functions
---------
var             : Value at Risk (historical, parametric, cornish-fisher).
cvar            : Conditional VaR / Expected Shortfall.
maxDrawdown     : Maximum drawdown from a price series.
sortino         : Sortino ratio.
calmar          : Calmar ratio.
hillEstimator   : Hill tail index estimator.
portfolioVar    : Portfolio VaR from a weight vector and return matrix.
rollingVol      : Rolling annualised volatility.
beta            : CAPM beta.
trackingError   : Tracking error and information ratio.
"""

import numpy as np


# ---------------------------------------------------------------------------
# VaR
# ---------------------------------------------------------------------------

def var(returns, alpha=0.05, method='historical'):
    """Value at Risk.

    Parameters
    ----------
    returns : (T,) return array.
    alpha   : float, significance level (e.g. 0.05 = 95% VaR).
    method  : 'historical' | 'parametric' | 'cornish_fisher'.

    Returns
    -------
    dict: var (positive = loss), method.
    """
    returns = np.asarray(returns, dtype=float)

    if method == 'historical':
        varVal = float(-np.percentile(returns, 100.0 * alpha))

    elif method == 'parametric':
        mu = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        # z-score for alpha quantile (standard normal).
        z = _normPpf(alpha)
        varVal = float(-(mu + z * sigma))

    elif method == 'cornish_fisher':
        mu = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        skew = float(_skewness(returns))
        kurt = float(_excessKurtosis(returns))
        z = _normPpf(alpha)
        # Cornish-Fisher expansion.
        zCF = (z
               + (z ** 2 - 1.0) * skew / 6.0
               + (z ** 3 - 3.0 * z) * kurt / 24.0
               - (2.0 * z ** 3 - 5.0 * z) * skew ** 2 / 36.0)
        varVal = float(-(mu + zCF * sigma))

    else:
        raise ValueError(f"Unknown method '{method}'. Use 'historical', 'parametric', or 'cornish_fisher'.")

    return {'var': varVal, 'method': method}


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------

def cvar(returns, alpha=0.05, method='historical'):
    """Conditional VaR (Expected Shortfall).

    Parameters
    ----------
    returns : (T,) return array.
    alpha   : float, significance level.
    method  : 'historical' | 'parametric'.

    Returns
    -------
    dict: cvar (positive = loss), var (positive = loss), method.
    """
    returns = np.asarray(returns, dtype=float)
    varResult = var(returns, alpha=alpha, method=method)
    varVal = varResult['var']

    if method == 'historical':
        threshold = -varVal  # returns below this are tail losses
        tailReturns = returns[returns <= threshold]
        if len(tailReturns) == 0:
            cvarVal = varVal
        else:
            cvarVal = float(-tailReturns.mean())

    elif method == 'parametric':
        mu = float(returns.mean())
        sigma = float(returns.std(ddof=1))
        z = _normPpf(alpha)
        phi_z = float(np.exp(-0.5 * z * z) / np.sqrt(2.0 * np.pi))
        cvarVal = float(-(mu - sigma * phi_z / (alpha + 1e-16)))

    else:
        # Fallback to historical.
        threshold = -varVal
        tailReturns = returns[returns <= threshold]
        cvarVal = float(-tailReturns.mean()) if len(tailReturns) > 0 else varVal

    return {'cvar': cvarVal, 'var': varVal, 'method': method}


# ---------------------------------------------------------------------------
# Max Drawdown
# ---------------------------------------------------------------------------

def maxDrawdown(prices):
    """Maximum drawdown from a price series.

    Parameters
    ----------
    prices : (T,) price array.

    Returns
    -------
    dict: maxDrawdown (negative or zero), peakIdx (int), troughIdx (int),
          drawdowns (T,).
    """
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    drawdowns = np.zeros(T)
    runningMax = prices[0]
    peakIdx = 0
    troughIdx = 0
    bestPeak = 0
    maxDD = 0.0

    for t in range(T):
        if prices[t] > runningMax:
            runningMax = prices[t]
            peakIdx = t
        dd = (prices[t] - runningMax) / (runningMax + 1e-16)
        drawdowns[t] = dd
        if dd < maxDD:
            maxDD = dd
            bestPeak = peakIdx
            troughIdx = t

    return {
        'maxDrawdown': float(maxDD),
        'peakIdx': int(bestPeak),
        'troughIdx': int(troughIdx),
        'drawdowns': drawdowns,
    }


# ---------------------------------------------------------------------------
# Sortino
# ---------------------------------------------------------------------------

def sortino(returns, rf=0.0, periodsPerYear=252):
    """Sortino ratio.

    Parameters
    ----------
    returns       : (T,) return array.
    rf            : float, risk-free rate (per period).
    periodsPerYear: int.

    Returns
    -------
    dict: sortino (float), annualReturn (float), downstdDev (float).
    """
    returns = np.asarray(returns, dtype=float)
    excess = returns - rf
    annualReturn = float(returns.mean() * periodsPerYear)
    downside = excess[excess < 0.0]
    if len(downside) == 0:
        downstdDev = 1e-16
    else:
        downstdDev = float(np.sqrt((downside ** 2).mean() * periodsPerYear))
    sortinoVal = (annualReturn - rf * periodsPerYear) / (downstdDev + 1e-16)

    return {
        'sortino': float(sortinoVal),
        'annualReturn': annualReturn,
        'downstdDev': float(downstdDev),
    }


# ---------------------------------------------------------------------------
# Calmar
# ---------------------------------------------------------------------------

def calmar(returns, periodsPerYear=252):
    """Calmar ratio.

    Parameters
    ----------
    returns       : (T,) return array.
    periodsPerYear: int.

    Returns
    -------
    dict: calmar (float), annualReturn (float), maxDrawdown (float).
    """
    returns = np.asarray(returns, dtype=float)
    annualReturn = float(returns.mean() * periodsPerYear)
    prices = np.cumprod(1.0 + returns)
    ddResult = maxDrawdown(prices)
    mdVal = ddResult['maxDrawdown']
    calmarVal = annualReturn / (abs(mdVal) + 1e-16)

    return {
        'calmar': float(calmarVal),
        'annualReturn': annualReturn,
        'maxDrawdown': float(mdVal),
    }


# ---------------------------------------------------------------------------
# Hill Estimator
# ---------------------------------------------------------------------------

def hillEstimator(returns, threshold=None):
    """Hill tail index estimator.

    Estimates the tail index xi of the distribution using the k largest
    absolute losses.

    Parameters
    ----------
    returns   : (T,) return array.
    threshold : float or None. If None, uses the 90th percentile of losses.

    Returns
    -------
    dict: xi (float, tail index), threshold (float), nExceedances (int).
    """
    returns = np.asarray(returns, dtype=float)
    losses = np.sort(-returns)[::-1]  # descending losses
    losses = losses[losses > 0]

    if threshold is None:
        threshold = float(np.percentile(losses, 90.0)) if len(losses) > 0 else 0.0

    exceedances = losses[losses > threshold]
    k = len(exceedances)

    if k < 2:
        return {'xi': float('nan'), 'threshold': float(threshold), 'nExceedances': k}

    xi = float(np.mean(np.log(exceedances / threshold)))

    return {'xi': xi, 'threshold': float(threshold), 'nExceedances': k}


# ---------------------------------------------------------------------------
# Portfolio VaR
# ---------------------------------------------------------------------------

def portfolioVar(weights, returns, alpha=0.05, method='historical'):
    """Portfolio VaR.

    Parameters
    ----------
    weights : (n,) weight vector.
    returns : (T x n) return matrix.
    alpha   : float.
    method  : 'historical' | 'parametric'.

    Returns
    -------
    dict: var (float), cvar (float), portfolioReturns (T,).
    """
    weights = np.asarray(weights, dtype=float)
    returns = np.asarray(returns, dtype=float)
    portRets = returns @ weights
    varResult = var(portRets, alpha=alpha, method=method)
    cvarResult = cvar(portRets, alpha=alpha, method=method)

    return {
        'var': varResult['var'],
        'cvar': cvarResult['cvar'],
        'portfolioReturns': portRets,
    }


# ---------------------------------------------------------------------------
# Rolling Volatility
# ---------------------------------------------------------------------------

def rollingVol(returns, window=21):
    """Rolling annualised volatility.

    Parameters
    ----------
    returns : (T,) return array.
    window  : int rolling window.

    Returns
    -------
    Array of length T - window + 1. Each entry: annualised std dev over window.
    """
    returns = np.asarray(returns, dtype=float)
    T = len(returns)
    nOut = T - window + 1
    out = np.zeros(nOut)
    for i in range(nOut):
        out[i] = float(returns[i:i + window].std(ddof=1)) * np.sqrt(252.0)
    return out


# ---------------------------------------------------------------------------
# Beta
# ---------------------------------------------------------------------------

def beta(assetReturns, marketReturns):
    """CAPM beta.

    Parameters
    ----------
    assetReturns  : (T,) asset return array.
    marketReturns : (T,) market return array.

    Returns
    -------
    dict: beta (float), alpha (float), rSquared (float).
    """
    assetReturns = np.asarray(assetReturns, dtype=float)
    marketReturns = np.asarray(marketReturns, dtype=float)

    xBar = float(marketReturns.mean())
    yBar = float(assetReturns.mean())
    cov = float(np.mean((marketReturns - xBar) * (assetReturns - yBar)))
    varM = float(np.mean((marketReturns - xBar) ** 2))
    betaVal = cov / (varM + 1e-16)
    alphaVal = yBar - betaVal * xBar

    # R-squared.
    predicted = alphaVal + betaVal * marketReturns
    ssTot = float(np.sum((assetReturns - yBar) ** 2))
    ssRes = float(np.sum((assetReturns - predicted) ** 2))
    rSq = 1.0 - ssRes / (ssTot + 1e-16)

    return {'beta': float(betaVal), 'alpha': float(alphaVal), 'rSquared': float(rSq)}


# ---------------------------------------------------------------------------
# Tracking Error
# ---------------------------------------------------------------------------

def trackingError(portfolioReturns, benchmarkReturns):
    """Tracking error and information ratio.

    Parameters
    ----------
    portfolioReturns  : (T,) portfolio return array.
    benchmarkReturns  : (T,) benchmark return array.

    Returns
    -------
    dict: trackingError (float, annualised), informationRatio (float),
          activeReturns (T,).
    """
    portfolioReturns = np.asarray(portfolioReturns, dtype=float)
    benchmarkReturns = np.asarray(benchmarkReturns, dtype=float)
    activeReturns = portfolioReturns - benchmarkReturns
    te = float(activeReturns.std(ddof=1)) * np.sqrt(252.0)
    ir = float(activeReturns.mean() * 252.0) / (te + 1e-16)

    return {
        'trackingError': float(te),
        'informationRatio': float(ir),
        'activeReturns': activeReturns,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normPpf(p, tol=1e-8, maxIter=50):
    """Inverse normal CDF via Newton-Raphson (Abramowitz & Stegun seed)."""
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    # Rational approximation seed (Abramowitz & Stegun 26.2.17).
    if p < 0.5:
        t = np.sqrt(-2.0 * np.log(p))
        sign = -1.0
    else:
        t = np.sqrt(-2.0 * np.log(1.0 - p))
        sign = 1.0

    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    num = c0 + c1 * t + c2 * t * t
    den = 1.0 + d1 * t + d2 * t * t + d3 * t ** 3
    x = sign * (t - num / den)

    # Newton-Raphson refinement.
    for _ in range(maxIter):
        fx = _normCdf(x) - p
        fpx = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
        dx = fx / (fpx + 1e-16)
        x -= dx
        if abs(dx) < tol:
            break
    return float(x)


def _normCdf(x):
    """Normal CDF (Abramowitz & Stegun polynomial approximation)."""
    x = float(x)
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    cdf = 1.0 - (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * x * x) * poly
    return cdf if x >= 0.0 else 1.0 - cdf


def _skewness(x):
    n = len(x)
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma < 1e-16:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 3))


def _excessKurtosis(x):
    n = len(x)
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma < 1e-16:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4)) - 3.0
