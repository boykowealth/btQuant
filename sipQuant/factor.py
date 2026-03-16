"""
sipQuant.factor — Factor models for asset returns.
SIP Global (Systematic Index Partners)

Pure NumPy. No external dependencies.

Functions
---------
capm         : CAPM regression: alpha, beta, R-squared, t-stats, p-values.
rollingBeta  : Rolling CAPM beta over a moving window.
pcaFactors   : PCA-based factor extraction from a returns panel.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _ols2(y, X):
    """OLS regression with a 2-column design matrix (no constant added).

    Parameters
    ----------
    y : (n,) array — dependent variable.
    X : (n, 2) array — design matrix (e.g., [constant, factor]).

    Returns
    -------
    beta      : (2,) coefficient array.
    residuals : (n,) residual array.
    """
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ beta
    return beta, residuals


def _tDist_pvalue(tStat, df):
    """Two-tailed p-value from t-distribution using a numerical approximation.

    Uses the regularised incomplete beta function approximation.
    For df > 0; returns p in (0, 1].
    """
    tStat = float(tStat)
    df = float(df)
    if df <= 0:
        return np.nan
    x = df / (df + tStat ** 2)
    # Regularised incomplete beta I_x(df/2, 1/2) via continued fraction (Lentz)
    a = df / 2.0
    b = 0.5
    # Compute I_x(a, b) using the simple series expansion for small x or large df
    # We use the relation to the F-distribution: p = I_{df/(df+t^2)}(df/2, 1/2)
    # Numerically safe approximation using beta CDF via recursion
    lnBeta = _lnBeta(a, b)
    ix = _regIncBeta(x, a, b, lnBeta)
    return float(np.clip(ix, 0.0, 1.0))


def _lnBeta(a, b):
    """log B(a, b) = log Gamma(a) + log Gamma(b) - log Gamma(a+b)."""
    return _lnGamma(a) + _lnGamma(b) - _lnGamma(a + b)


def _lnGamma(z):
    """Lanczos approximation to log Gamma(z) for z > 0."""
    g = 7
    c = [0.99999999999980993, 676.5203681218851, -1259.1392167224028,
         771.32342877765313, -176.61502916214059, 12.507343278686905,
         -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7]
    if z < 0.5:
        return np.log(np.pi) - np.log(np.abs(np.sin(np.pi * z))) - _lnGamma(1.0 - z)
    z -= 1
    x = c[0]
    for i in range(1, g + 2):
        x += c[i] / (z + i)
    t = z + g + 0.5
    return 0.5 * np.log(2.0 * np.pi) + (z + 0.5) * np.log(t) - t + np.log(x)


def _regIncBeta(x, a, b, lnBeta):
    """Regularised incomplete beta function I_x(a,b) via continued fraction."""
    if x < 0.0 or x > 1.0:
        return np.nan
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    lnX = np.log(x)
    lnOneMinusX = np.log(1.0 - x)
    lnFront = a * lnX + b * lnOneMinusX - lnBeta - np.log(a)

    # Use symmetry for faster convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regIncBeta(1.0 - x, b, a, lnBeta)

    # Continued fraction via Lentz method
    fpMin = 1e-30
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpMin:
        d = fpMin
    d = 1.0 / d
    h = d
    for m in range(1, 101):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpMin:
            d = fpMin
        c = 1.0 + aa / c
        if abs(c) < fpMin:
            c = fpMin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpMin:
            d = fpMin
        c = 1.0 + aa / c
        if abs(c) < fpMin:
            c = fpMin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 3e-7:
            break
    return np.exp(lnFront) * h


# ---------------------------------------------------------------------------
# CAPM
# ---------------------------------------------------------------------------

def capm(assetReturns, marketReturns, rf=0.0):
    """CAPM regression: excess asset return = alpha + beta * excess market return.

    Parameters
    ----------
    assetReturns  : array-like, length n.
    marketReturns : array-like, length n.
    rf            : float — risk-free rate (scalar, same units as returns).

    Returns
    -------
    dict with keys: alpha, beta, rSquared, tStatAlpha, tStatBeta,
                    pValueAlpha, pValueBeta.
    """
    assetReturns = np.asarray(assetReturns, dtype=float)
    marketReturns = np.asarray(marketReturns, dtype=float)
    n = len(assetReturns)

    yExcess = assetReturns - rf
    xExcess = marketReturns - rf

    X = np.column_stack([np.ones(n), xExcess])
    beta, residuals = _ols2(yExcess, X)
    alpha, betaVal = float(beta[0]), float(beta[1])

    # R-squared
    ssTot = float(np.sum((yExcess - np.mean(yExcess)) ** 2))
    ssRes = float(np.sum(residuals ** 2))
    rSquared = 1.0 - ssRes / ssTot if ssTot > 0 else 0.0

    # Standard errors
    df = n - 2
    s2 = ssRes / df if df > 0 else np.nan
    XtXinv = np.linalg.pinv(X.T @ X)
    seAlpha = float(np.sqrt(s2 * XtXinv[0, 0]))
    seBeta = float(np.sqrt(s2 * XtXinv[1, 1]))

    tStatAlpha = alpha / seAlpha if seAlpha > 0 else np.nan
    tStatBeta = betaVal / seBeta if seBeta > 0 else np.nan

    pValueAlpha = _tDist_pvalue(tStatAlpha, df)
    pValueBeta = _tDist_pvalue(tStatBeta, df)

    return {
        'alpha': alpha,
        'beta': betaVal,
        'rSquared': float(rSquared),
        'tStatAlpha': float(tStatAlpha),
        'tStatBeta': float(tStatBeta),
        'pValueAlpha': float(pValueAlpha),
        'pValueBeta': float(pValueBeta),
    }


# ---------------------------------------------------------------------------
# Rolling beta
# ---------------------------------------------------------------------------

def rollingBeta(assetReturns, marketReturns, window=60, rf=0.0):
    """Rolling CAPM beta over a sliding window.

    Parameters
    ----------
    assetReturns  : array-like, length n.
    marketReturns : array-like, length n.
    window        : int — rolling window length.
    rf            : float — risk-free rate.

    Returns
    -------
    dict with keys: betas (array, length n - window + 1),
                    alphas (array), rSquared (array).
    """
    assetReturns = np.asarray(assetReturns, dtype=float)
    marketReturns = np.asarray(marketReturns, dtype=float)
    n = len(assetReturns)
    window = int(window)

    nWindows = n - window + 1
    betas = np.empty(nWindows)
    alphas = np.empty(nWindows)
    rSquaredArr = np.empty(nWindows)

    for i in range(nWindows):
        ar = assetReturns[i: i + window]
        mr = marketReturns[i: i + window]
        result = capm(ar, mr, rf=rf)
        betas[i] = result['beta']
        alphas[i] = result['alpha']
        rSquaredArr[i] = result['rSquared']

    return {
        'betas': betas,
        'alphas': alphas,
        'rSquared': rSquaredArr,
    }


# ---------------------------------------------------------------------------
# PCA factors
# ---------------------------------------------------------------------------

def pcaFactors(returns, nFactors=3):
    """Extract PCA-based factors from a returns panel.

    Demeans the returns, performs SVD, and computes factor returns,
    loadings, and per-asset R-squared from the factor model fit.

    Parameters
    ----------
    returns  : array-like, shape (T, n) — T time periods, n assets.
    nFactors : int — number of PCA factors to extract.

    Returns
    -------
    dict with keys:
        factors           — (T, nFactors) factor time series.
        loadings          — (n, nFactors) factor loadings.
        explainedVariance — (nFactors,) variance explained per factor.
        factorReturns     — (T, nFactors) alias for factors.
        residuals         — (T, n) idiosyncratic residuals.
        rSquaredByAsset   — (n,) per-asset R-squared from factor model fit.
    """
    returns = np.asarray(returns, dtype=float)
    T, n = returns.shape
    nFactors = int(nFactors)

    # Demean
    Rc = returns - np.mean(returns, axis=0)

    # SVD
    U, s, Vt = np.linalg.svd(Rc, full_matrices=False)

    explainedVariance = (s ** 2) / (T - 1)

    # Factor returns: (T, nFactors), loadings: (n, nFactors)
    factors = U[:, :nFactors] * s[:nFactors]         # (T, nFactors)
    loadings = Vt[:nFactors].T                        # (n, nFactors)

    # Reconstruct factor-model fitted values
    fitted = factors @ loadings.T                      # (T, n)
    residuals = Rc - fitted

    # Per-asset R-squared
    ssTotByAsset = np.sum(Rc ** 2, axis=0)
    ssResByAsset = np.sum(residuals ** 2, axis=0)
    rSquaredByAsset = np.where(
        ssTotByAsset > 0,
        1.0 - ssResByAsset / ssTotByAsset,
        0.0,
    )

    return {
        'factors': factors,
        'loadings': loadings,
        'explainedVariance': explainedVariance[:nFactors],
        'factorReturns': factors,
        'residuals': residuals,
        'rSquaredByAsset': rSquaredByAsset,
    }
