import numpy as np


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _tCdfScalar(x, df):
    """Student's t CDF for a scalar x."""
    a = df / (df + x**2)
    return 1.0 - 0.5 * _betaInc(df / 2, 0.5, a)


def _tCdf(x, df):
    """Student's t CDF — handles scalar or array x."""
    x = np.asarray(x, dtype=float)
    scalar_input = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.array([_tCdfScalar(xi, df) for xi in x.flat]).reshape(x.shape)
    return float(result[0]) if scalar_input else result


def _betaInc(a, b, x):
    """Incomplete beta function approximation."""
    if x <= 0:
        return 0
    if x >= 1:
        return 1

    bt = np.exp(a * np.log(x) + b * np.log(1 - x) - _logBeta(a, b))

    if x < (a + 1) / (a + b + 2):
        return bt * _betaCf(a, b, x) / a
    else:
        return 1 - bt * _betaCf(b, a, 1 - x) / b


def _betaCf(a, b, x, maxIter=100):
    """Continued fraction for incomplete beta function."""
    qab = a + b
    qap = a + 1
    qam = a - 1
    c = 1.0
    d = 1.0 - qab * x / qap

    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d

    for m in range(1, maxIter):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < 1e-8:
            break

    return h


def _logBeta(a, b):
    """Log beta function."""
    return _logGamma(a) + _logGamma(b) - _logGamma(a + b)


def _logGamma(x):
    """Log gamma function approximation."""
    if x <= 0:
        return np.inf

    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5]

    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * np.log(tmp)
    ser = 1.000000000190015

    for c in cof:
        y += 1
        ser += c / y

    return -tmp + np.log(2.5066282746310005 * ser / x)


def _chiSqCdf(x, df):
    """Chi-squared cumulative distribution function."""
    return _gammaInc(df / 2, x / 2)


def _gammaInc(a, x):
    """Incomplete gamma function."""
    if x < 0 or a <= 0:
        return 0.0

    if x < a + 1:
        ap = a
        delta = 1.0 / a
        sumVal = delta
        for n in range(1, 100):
            ap += 1
            delta *= x / ap
            sumVal += delta
            if delta < sumVal * 1e-10:
                break
        return sumVal * np.exp(-x + a * np.log(x) - _logGamma(a))
    else:
        b = x + 1 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, 100):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            delta = d * c
            h *= delta
            if abs(delta - 1.0) < 1e-10:
                break
        return 1.0 - h * np.exp(-x + a * np.log(x) - _logGamma(a))


def _chiSqInv(p, df):
    """Inverse chi-squared distribution."""
    if p <= 0 or p >= 1:
        return np.nan

    x = df
    for _ in range(50):
        cdf = _chiSqCdf(x, df)
        pdf = (x ** (df / 2 - 1) * np.exp(-x / 2)) / (2 ** (df / 2) * np.exp(_logGamma(df / 2)))
        xNew = x - (cdf - p) / pdf
        if abs(xNew - x) < 1e-8:
            return xNew
        x = xNew

    return x


# ---------------------------------------------------------------------------
# Existing functions
# ---------------------------------------------------------------------------

def ols(y, X, addConst=True, robust=False, covType='HC3'):
    """
    Ordinary least squares regression with optional robust standard errors.

    Parameters
    ----------
    y : array_like
        Dependent variable (1-D).
    X : array_like
        Independent variables (2-D, or 1-D for a single predictor).
    addConst : bool, optional
        Prepend a column of ones. Default True.
    robust : bool, optional
        Use heteroskedasticity-robust standard errors. Default False.
    covType : str, optional
        HC sandwich type: 'HC0', 'HC1', 'HC2', 'HC3', 'HC4'. Default 'HC3'.

    Returns
    -------
    dict
        coefficients, stdErrors, tStats, pValues, confIntLower, confIntUpper,
        residuals, fittedValues, rSquared, adjRSquared, fStat, fPValue,
        nObs, df, sse, covMatrix
    """
    y = np.asarray(y).flatten()
    X = np.asarray(X)

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    if addConst:
        X = np.column_stack([np.ones(len(y)), X])

    n, k = X.shape

    XtXInv = np.linalg.inv(X.T @ X)
    beta = XtXInv @ X.T @ y

    residuals = y - X @ beta
    yHat = X @ beta

    df = n - k

    SSR = np.sum((yHat - np.mean(y))**2)
    SST = np.sum((y - np.mean(y))**2)
    SSE = np.sum(residuals**2)

    rSquared = SSR / SST
    adjRSquared = 1 - (1 - rSquared) * (n - 1) / df

    if not robust:
        sigma2 = SSE / df
        covMatrix = sigma2 * XtXInv
        stdErrors = np.sqrt(np.diag(covMatrix))
    else:
        if covType == 'HC0':
            u2 = residuals**2
        elif covType == 'HC1':
            u2 = residuals**2 * (n / df)
        elif covType == 'HC2':
            h = np.diag(X @ XtXInv @ X.T)
            u2 = residuals**2 / (1 - h)
        elif covType == 'HC3':
            h = np.diag(X @ XtXInv @ X.T)
            u2 = residuals**2 / (1 - h)**2
        elif covType == 'HC4':
            h = np.diag(X @ XtXInv @ X.T)
            delta = np.minimum(4, h / np.mean(h))
            u2 = residuals**2 / (1 - h)**delta
        else:
            raise ValueError("covType must be HC0, HC1, HC2, HC3, or HC4")

        Xu2 = X * u2[:, np.newaxis]
        covMatrix = XtXInv @ (X.T @ Xu2) @ XtXInv
        stdErrors = np.sqrt(np.diag(covMatrix))

    tStats = beta / stdErrors
    pValues = 2 * (1 - _tCdf(np.abs(tStats), df))

    tCrit = 1.96
    ciLower = beta - tCrit * stdErrors
    ciUpper = beta + tCrit * stdErrors

    fStat = None
    fPValue = None
    if addConst and k > 1:
        rss1 = SSE
        XRestricted = X[:, 0].reshape(-1, 1)
        betaRestricted = np.linalg.inv(XRestricted.T @ XRestricted) @ XRestricted.T @ y
        residualsRestricted = y - XRestricted @ betaRestricted
        rss0 = np.sum(residualsRestricted**2)

        fStat = ((rss0 - rss1) / (k - 1)) / (rss1 / df)
        fPValue = 1 - _chiSqCdf(fStat * (k - 1), k - 1)

    return {
        'coefficients': beta,
        'stdErrors': stdErrors,
        'tStats': tStats,
        'pValues': pValues,
        'confIntLower': ciLower,
        'confIntUpper': ciUpper,
        'residuals': residuals,
        'fittedValues': yHat,
        'rSquared': rSquared,
        'adjRSquared': adjRSquared,
        'fStat': fStat,
        'fPValue': fPValue,
        'nObs': n,
        'df': df,
        'sse': SSE,
        'covMatrix': covMatrix,
    }


def whiteTest(y, X, addConst=True):
    """
    White's test for heteroskedasticity.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    X : array_like
        Independent variables.
    addConst : bool, optional
        Add constant term. Default True.

    Returns
    -------
    dict
        testStatistic, pValue, df, conclusion
    """
    results = ols(y, X, addConst=addConst)
    resid = results['residuals']
    residSq = resid**2

    XOrig = np.asarray(X)
    if XOrig.ndim == 1:
        XOrig = XOrig.reshape(-1, 1)

    n, k = XOrig.shape

    XWhite = [XOrig[:, i] for i in range(k)]
    XWhite.extend([XOrig[:, i]**2 for i in range(k)])

    for i in range(k):
        for j in range(i + 1, k):
            XWhite.append(XOrig[:, i] * XOrig[:, j])

    XWhite = np.column_stack(XWhite)

    auxResults = ols(residSq, XWhite, addConst=True)

    testStat = n * auxResults['rSquared']
    dfTest = XWhite.shape[1]
    pValue = 1 - _chiSqCdf(testStat, dfTest)

    conclusion = "Reject H0: homoskedasticity" if pValue < 0.05 else "Fail to reject H0"

    return {
        'testStatistic': testStat,
        'pValue': pValue,
        'df': dfTest,
        'conclusion': conclusion,
    }


def breuschPaganTest(y, X, addConst=True):
    """
    Breusch-Pagan test for heteroskedasticity.

    Parameters
    ----------
    y : array_like
        Dependent variable.
    X : array_like
        Independent variables.
    addConst : bool, optional
        Add constant term. Default True.

    Returns
    -------
    dict
        testStatistic, pValue, df, conclusion
    """
    results = ols(y, X, addConst=addConst)
    resid = results['residuals']

    n = len(resid)
    sigma2 = np.sum(resid**2) / n
    residSqNorm = resid**2 / sigma2

    bpResults = ols(residSqNorm, X, addConst=addConst)

    explainedSS = bpResults['rSquared'] * np.sum((residSqNorm - np.mean(residSqNorm))**2)
    testStat = 0.5 * explainedSS

    XArr = np.asarray(X)
    dfTest = XArr.shape[1] if XArr.ndim > 1 else 1
    pValue = 1 - _chiSqCdf(testStat, dfTest)

    conclusion = "Reject H0: homoskedasticity" if pValue < 0.05 else "Fail to reject H0"

    return {
        'testStatistic': testStat,
        'pValue': pValue,
        'df': dfTest,
        'conclusion': conclusion,
    }


def durbinWatson(residuals):
    """
    Durbin-Watson statistic for autocorrelation.

    DW ~2 indicates no autocorrelation; 0 < DW < 2 positive; 2 < DW < 4 negative.

    Parameters
    ----------
    residuals : array_like
        Residuals from a regression model.

    Returns
    -------
    float
        Durbin-Watson statistic.
    """
    residuals = np.asarray(residuals)
    diffSquared = np.sum(np.diff(residuals)**2)
    residSquared = np.sum(residuals**2)
    return diffSquared / residSquared


def ljungBox(residuals, lags=None):
    """
    Ljung-Box test for autocorrelation in residuals.

    Parameters
    ----------
    residuals : array_like
        Residuals from a regression or time-series model.
    lags : int, optional
        Number of lags. Defaults to min(10, n // 5).

    Returns
    -------
    dict
        lags, autocorrelations, qStats, pValues, criticalValues
    """
    residuals = np.asarray(residuals)
    n = len(residuals)

    if lags is None:
        lags = min(10, n // 5)

    acfValues = []
    for k in range(1, lags + 1):
        numerator = np.sum(residuals[k:] * residuals[:-k])
        denominator = np.sum(residuals**2)
        acfValues.append(numerator / denominator)

    acfValues = np.array(acfValues)

    qStats = []
    pValues = []

    for l in range(1, lags + 1):
        q = n * (n + 2) * np.sum((acfValues[:l]**2) / (n - np.arange(1, l + 1)))
        qStats.append(q)
        pValues.append(1 - _chiSqCdf(q, l))

    critVals = [_chiSqInv(0.95, i) for i in range(1, lags + 1)]

    return {
        'lags': list(range(1, lags + 1)),
        'autocorrelations': acfValues,
        'qStats': qStats,
        'pValues': pValues,
        'criticalValues': critVals,
    }


def adfTest(series, lags=None, regression='c', autolag='AIC'):
    """
    Augmented Dickey-Fuller test for a unit root.

    Parameters
    ----------
    series : array_like
        Time series to test.
    lags : int, optional
        Number of lags. Chosen automatically via *autolag* if None.
    regression : str, optional
        'nc' no constant, 'c' constant, 'ct' constant + trend. Default 'c'.
    autolag : str, optional
        Information criterion for lag selection: 'AIC' or 'BIC'. Default 'AIC'.

    Returns
    -------
    dict
        testStatistic, pValue, criticalValues, conclusion, optimalLag
    """
    series = np.asarray(series)
    n = len(series)

    if lags is None:
        maxLags = int(np.ceil(12 * (n / 100)**(1 / 4)))
    else:
        maxLags = lags

    dy = np.diff(series)
    y1 = series[:-1]

    if lags is None and autolag is not None:
        results = {}
        for p in range(maxLags + 1):
            X = []

            if regression == 'c':
                X.append(np.ones(n - 1))
            elif regression == 'ct':
                X.append(np.ones(n - 1))
                X.append(np.arange(1, n))

            X.append(y1)

            for i in range(1, p + 1):
                lag = np.zeros(n - 1)
                lag[i:] = dy[:-i]
                X.append(lag)

            if not X:
                X = np.zeros((n - 1, 0))
            else:
                X = np.column_stack(X)

            model = ols(dy, X, addConst=False)

            k = X.shape[1]
            ssr = np.sum(model['residuals']**2)

            if autolag == 'AIC':
                ic = np.log(ssr / (n - 1)) + 2 * k / (n - 1)
            else:
                ic = np.log(ssr / (n - 1)) + k * np.log(n - 1) / (n - 1)

            results[p] = {'ic': ic, 'model': model, 'X': X}

        optimalLag = min(results.keys(), key=lambda x: results[x]['ic'])
        model = results[optimalLag]['model']
    else:
        p = maxLags
        X = []

        if regression == 'c':
            X.append(np.ones(n - 1))
        elif regression == 'ct':
            X.append(np.ones(n - 1))
            X.append(np.arange(1, n))

        X.append(y1)

        for i in range(1, p + 1):
            lag = np.zeros(n - 1)
            lag[i:] = dy[:-i]
            X.append(lag)

        if not X:
            X = np.zeros((n - 1, 0))
        else:
            X = np.column_stack(X)

        model = ols(dy, X, addConst=False)
        optimalLag = maxLags

    idx = 0
    if regression == 'c':
        idx = 1
    elif regression == 'ct':
        idx = 2

    adfStat = model['tStats'][idx]

    if regression == 'nc':
        critVals = {'1%': -2.58, '5%': -1.95, '10%': -1.62}
    elif regression == 'c':
        critVals = {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    else:
        critVals = {'1%': -3.96, '5%': -3.41, '10%': -3.13}

    pValue = 0.05

    conclusion = "Reject H0: unit root" if adfStat < critVals['5%'] else "Fail to reject H0"

    return {
        'testStatistic': adfStat,
        'pValue': pValue,
        'criticalValues': critVals,
        'conclusion': conclusion,
        'optimalLag': optimalLag,
    }


def kpssTest(series, lags=None, regression='c'):
    """
    KPSS test for stationarity.

    Parameters
    ----------
    series : array_like
        Time series to test.
    lags : int, optional
        Bandwidth for long-run variance. Auto-selected if None.
    regression : str, optional
        'c' constant or 'ct' constant + trend. Default 'c'.

    Returns
    -------
    dict
        testStatistic, pValue, criticalValues, conclusion, lags
    """
    series = np.asarray(series)
    n = len(series)

    if lags is None:
        lags = int(np.ceil(12 * (n / 100)**(1 / 4)))

    if regression == 'c':
        resid = series - np.mean(series)
    else:
        t = np.arange(1, n + 1)
        X = np.column_stack([np.ones(n), t])
        beta = np.linalg.lstsq(X, series, rcond=None)[0]
        resid = series - X @ beta

    s = np.cumsum(resid)

    gamma0 = np.sum(resid**2) / n

    autoCov = np.zeros(lags + 1)
    autoCov[0] = gamma0

    for l in range(1, lags + 1):
        autoCov[l] = np.sum(resid[l:] * resid[:-l]) / n

    w = 1 - np.arange(1, lags + 1) / (lags + 1)

    s2 = gamma0 + 2 * np.sum(w * autoCov[1:])

    kpssStat = np.sum(s**2) / (n**2 * s2)

    if regression == 'c':
        critVals = {'1%': 0.739, '5%': 0.463, '10%': 0.347}
    else:
        critVals = {'1%': 0.216, '5%': 0.146, '10%': 0.119}

    pValue = 0.05

    conclusion = "Reject H0: stationarity" if kpssStat > critVals['5%'] else "Fail to reject H0"

    return {
        'testStatistic': kpssStat,
        'pValue': pValue,
        'criticalValues': critVals,
        'conclusion': conclusion,
        'lags': lags,
    }


def grangerCausality(y, x, maxLag=4):
    """
    Granger causality test: does x Granger-cause y?

    Parameters
    ----------
    y : array_like
        Dependent time series.
    x : array_like
        Candidate causal time series.
    maxLag : int, optional
        Maximum number of lags to test. Default 4.

    Returns
    -------
    dict
        lags, fStats, pValues, conclusion
    """
    y = np.asarray(y)
    x = np.asarray(x)

    n = len(y)
    fStats = []
    pValues = []

    for lag in range(1, maxLag + 1):
        yLagged = []
        xLagged = []

        for i in range(1, lag + 1):
            yLagged.append(y[lag - i:n - i])
            xLagged.append(x[lag - i:n - i])

        yTrain = y[lag:]

        XRestricted = np.column_stack(yLagged)
        XUnrestricted = np.column_stack(yLagged + xLagged)

        modelRestricted = ols(yTrain, XRestricted, addConst=True)
        modelUnrestricted = ols(yTrain, XUnrestricted, addConst=True)

        rss1 = modelRestricted['sse']
        rss2 = modelUnrestricted['sse']

        dfNum = lag
        dfDenom = n - lag - 2 * lag - 1

        if dfDenom > 0 and rss2 > 0:
            fStat = ((rss1 - rss2) / dfNum) / (rss2 / dfDenom)
            pValue = 1 - _chiSqCdf(fStat * dfNum, dfNum)
        else:
            fStat = np.nan
            pValue = np.nan

        fStats.append(fStat)
        pValues.append(pValue)

    minPValue = np.nanmin(pValues) if len(pValues) > 0 else 1.0
    conclusion = "x Granger-causes y" if minPValue < 0.05 else "No Granger causality"

    return {
        'lags': list(range(1, maxLag + 1)),
        'fStats': fStats,
        'pValues': pValues,
        'conclusion': conclusion,
    }


# ---------------------------------------------------------------------------
# New functions
# ---------------------------------------------------------------------------

def huberRegression(y, X, addConst=True, delta=1.345, maxIter=100, tol=1e-6):
    """
    Huber regression via iteratively reweighted least squares (IRLS).

    Parameters
    ----------
    y : array_like
        Dependent variable (1-D).
    X : array_like
        Independent variables.
    addConst : bool, optional
        Prepend a column of ones. Default True.
    delta : float, optional
        Huber threshold parameter. Default 1.345.
    maxIter : int, optional
        Maximum IRLS iterations. Default 100.
    tol : float, optional
        Convergence tolerance on coefficient change. Default 1e-6.

    Returns
    -------
    dict
        coefficients, stdErrors, tStats, pValues, residuals, fittedValues,
        rSquared, nObs, df
    """
    y = np.asarray(y, dtype=float).flatten()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if addConst:
        X = np.column_stack([np.ones(len(y)), X])

    n, k = X.shape

    # Initialise with OLS
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(maxIter):
        resid = y - X @ beta

        # Robust scale estimate via MAD
        mad = np.median(np.abs(resid - np.median(resid)))
        sigma = mad / 0.6745 if mad > 0 else 1.0

        # Huber weights
        u = resid / sigma
        w = np.where(np.abs(u) <= delta, 1.0, delta / np.abs(u))

        # Weighted least squares
        W = np.diag(w)
        XtW = X.T @ W
        betaNew = np.linalg.solve(XtW @ X, XtW @ y)

        if np.max(np.abs(betaNew - beta)) < tol:
            beta = betaNew
            break
        beta = betaNew

    resid = y - X @ beta
    yHat = X @ beta
    df = n - k

    SST = np.sum((y - np.mean(y))**2)
    SSE = np.sum(resid**2)
    rSquared = 1 - SSE / SST if SST > 0 else 0.0

    sigma2 = SSE / df if df > 0 else 0.0
    XtXInv = np.linalg.pinv(X.T @ X)
    stdErrors = np.sqrt(np.maximum(np.diag(sigma2 * XtXInv), 0))

    tStats = beta / np.where(stdErrors > 0, stdErrors, np.nan)
    pValues = 2 * (1 - _tCdf(np.abs(tStats), df))

    return {
        'coefficients': beta,
        'stdErrors': stdErrors,
        'tStats': tStats,
        'pValues': pValues,
        'residuals': resid,
        'fittedValues': yHat,
        'rSquared': rSquared,
        'nObs': n,
        'df': df,
    }


def theilSen(y, X):
    """
    Theil-Sen slope estimator.

    For a single predictor the slope is the median of all pairwise slopes.
    For multiple predictors each column is handled independently and the
    intercept is set via the median of (y - X @ slopes).

    Parameters
    ----------
    y : array_like
        Dependent variable (1-D, length n).
    X : array_like
        Independent variable(s). 1-D or 2-D (n x p).

    Returns
    -------
    dict
        coefficients (intercept prepended), intercept, slopes
    """
    y = np.asarray(y, dtype=float).flatten()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n, p = X.shape
    slopes = np.zeros(p)

    for col in range(p):
        xc = X[:, col]
        pairSlopes = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = xc[j] - xc[i]
                if dx != 0:
                    pairSlopes.append((y[j] - y[i]) / dx)
        slopes[col] = np.median(pairSlopes) if pairSlopes else 0.0

    intercept = np.median(y - X @ slopes)
    coefficients = np.concatenate([[intercept], slopes])

    return {
        'coefficients': coefficients,
        'intercept': intercept,
        'slopes': slopes,
    }


def madVol(returns, window=None):
    """
    MAD-based robust volatility: 1.4826 * median(|r - median(r)|).

    Parameters
    ----------
    returns : array_like
        Return series.
    window : int, optional
        Rolling window length. If None a single scalar is returned.

    Returns
    -------
    dict
        vol : float or ndarray
            Scalar when *window* is None; rolling array of length
            ``n - window + 1`` otherwise.
    """
    r = np.asarray(returns, dtype=float).flatten()

    if window is None:
        vol = 1.4826 * np.median(np.abs(r - np.median(r)))
        return {'vol': vol}

    n = len(r)
    out = np.empty(n - window + 1)
    for i in range(len(out)):
        seg = r[i:i + window]
        out[i] = 1.4826 * np.median(np.abs(seg - np.median(seg)))

    return {'vol': out}


def chowTest(y, X, breakPoint, addConst=True):
    """
    Chow structural break test.

    Splits the sample at *breakPoint* and tests whether pooled OLS fits
    significantly worse than the two separate regressions.

    Parameters
    ----------
    y : array_like
        Dependent variable (1-D, length n).
    X : array_like
        Independent variables.
    breakPoint : int
        Index splitting the sample into ``[0:breakPoint]`` and ``[breakPoint:]``.
    addConst : bool, optional
        Add constant term. Default True.

    Returns
    -------
    dict
        fStat, pValue, conclusion
    """
    y = np.asarray(y, dtype=float).flatten()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = len(y)
    k = X.shape[1] + (1 if addConst else 0)

    # Pooled
    try:
        resPooled = ols(y, X, addConst=addConst)
        ssrPooled = resPooled['sse']
        res1 = ols(y[:breakPoint], X[:breakPoint], addConst=addConst)
        res2 = ols(y[breakPoint:], X[breakPoint:], addConst=addConst)
        ssrUnrestricted = res1['sse'] + res2['sse']
    except np.linalg.LinAlgError:
        return {'fStat': np.nan, 'pValue': np.nan,
                'conclusion': 'Fail to reject H0 (singular matrix)'}

    dfNum = k
    dfDenom = n - 2 * k

    if dfDenom <= 0 or ssrUnrestricted <= 0:
        fStat = np.nan
        pValue = np.nan
    else:
        fStat = ((ssrPooled - ssrUnrestricted) / dfNum) / (ssrUnrestricted / dfDenom)
        pValue = 1 - _chiSqCdf(fStat * dfNum, dfNum)

    conclusion = "Reject H0: structural break detected" if (not np.isnan(pValue) and pValue < 0.05) else "Fail to reject H0"

    return {
        'fStat': fStat,
        'pValue': pValue,
        'conclusion': conclusion,
    }


def baiPerron(y, X, maxBreaks=3, addConst=True):
    """
    Bai-Perron multiple structural break test via grid search.

    For each candidate number of breaks ``b = 1 .. maxBreaks`` all
    combinations of break indices (with minimum segment length
    ``max(15, n // 10)``) are evaluated; the combination minimising
    total SSR is retained.

    Parameters
    ----------
    y : array_like
        Dependent variable (1-D).
    X : array_like
        Independent variables.
    maxBreaks : int, optional
        Maximum number of structural breaks to consider. Default 3.
    addConst : bool, optional
        Add constant term. Default True.

    Returns
    -------
    dict
        nBreaks, breakIndices, ssrByBreaks
    """
    y = np.asarray(y, dtype=float).flatten()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n = len(y)
    minSeg = max(15, n // 10)

    def segSsr(start, end):
        if end - start < 2:
            return np.sum((y[start:end] - np.mean(y[start:end]))**2) if end > start else 0.0
        r = ols(y[start:end], X[start:end], addConst=addConst)
        return r['sse']

    ssrByBreaks = {}
    bestBreaks = 0
    bestIndices = []
    bestSsr = segSsr(0, n)

    from itertools import combinations

    for b in range(1, maxBreaks + 1):
        candidates = range(minSeg, n - minSeg)
        if len(list(candidates)) < b:
            continue

        bestSsrB = np.inf
        bestIdxB = None

        for bkpts in combinations(candidates, b):
            # Enforce minimum segment length between consecutive breaks
            pts = list(bkpts)
            pts_ext = [0] + pts + [n]
            if any(pts_ext[i + 1] - pts_ext[i] < minSeg for i in range(len(pts_ext) - 1)):
                continue

            totalSsr = sum(segSsr(pts_ext[i], pts_ext[i + 1]) for i in range(len(pts_ext) - 1))
            if totalSsr < bestSsrB:
                bestSsrB = totalSsr
                bestIdxB = pts

        if bestIdxB is not None:
            ssrByBreaks[b] = bestSsrB
            if bestSsrB < bestSsr:
                bestSsr = bestSsrB
                bestBreaks = b
                bestIndices = bestIdxB

    ssrByBreaks[0] = segSsr(0, n)

    return {
        'nBreaks': bestBreaks,
        'breakIndices': bestIndices,
        'ssrByBreaks': ssrByBreaks,
    }


def cusum(y, X, addConst=True):
    """
    CUSUM test for structural change based on recursive residuals.

    The cumulative sum of standardised recursive residuals is computed
    from an expanding OLS. The 5 % critical bands follow the standard
    Brown-Durbin-Evans boundary.

    Parameters
    ----------
    y : array_like
        Dependent variable (1-D).
    X : array_like
        Independent variables.
    addConst : bool, optional
        Add constant term. Default True.

    Returns
    -------
    dict
        cusumStat (ndarray), critBands (dict with 'upper' and 'lower'),
        conclusion
    """
    y = np.asarray(y, dtype=float).flatten()
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if addConst:
        X = np.column_stack([np.ones(len(y)), X])

    n, k = X.shape
    startIdx = k + 1  # minimum observations for first OLS

    recResid = []
    for t in range(startIdx, n + 1):
        Xt = X[:t - 1]
        yt = y[:t - 1]
        beta = np.linalg.lstsq(Xt, yt, rcond=None)[0]

        xNew = X[t - 1]
        yPred = xNew @ beta
        eNew = y[t - 1] - yPred

        # Prediction variance
        XtXInv = np.linalg.pinv(Xt.T @ Xt)
        sigmaHat2 = np.sum((yt - Xt @ beta)**2) / max(t - 1 - k, 1)
        fVar = sigmaHat2 * (1 + xNew @ XtXInv @ xNew)
        recResid.append(eNew / np.sqrt(max(fVar, 1e-12)))

    recResid = np.array(recResid)
    sigma = np.std(recResid, ddof=1) if len(recResid) > 1 else 1.0
    cusumStat = np.cumsum(recResid) / (sigma * np.sqrt(len(recResid)))

    m = len(cusumStat)
    t_arr = np.arange(1, m + 1)
    # 5 % critical band: ±(0.948 + 2*0.948*(t/m - 0.5)) scaled
    upper = 0.948 * (1 + 2 * t_arr / m)
    lower = -upper

    maxStat = np.max(np.abs(cusumStat))
    conclusion = "Reject H0: structural change detected" if maxStat > 0.948 else "Fail to reject H0"

    return {
        'cusumStat': cusumStat,
        'critBands': {'upper': upper, 'lower': lower},
        'conclusion': conclusion,
    }
