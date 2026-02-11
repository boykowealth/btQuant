import numpy as np

def _tCdf(x, df):
    """Student's t cumulative distribution function."""
    a = df / (df + x**2)
    return 1.0 - 0.5 * _betaInc(df / 2, 0.5, a)

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

def ols(y, X, addConst=True, robust=False, covType='HC3'):
    """
    Ordinary least squares regression with optional robust standard errors.
    
    Parameters:
        y: dependent variable (1D array)
        X: independent variables (2D array or 1D for single predictor)
        addConst: add constant term
        robust: use robust standard errors
        covType: 'HC0', 'HC1', 'HC2', 'HC3', 'HC4'
    
    Returns:
        dict with coefficients, std_errors, t_stats, p_values, r_squared, etc.
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
        'covMatrix': covMatrix
    }

def whiteTest(y, X, addConst=True):
    """
    White's test for heteroskedasticity.
    
    Parameters:
        y: dependent variable
        X: independent variables
        addConst: add constant term
    
    Returns:
        dict with testStatistic, pValue, df, conclusion
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
        'conclusion': conclusion
    }

def breuschPaganTest(y, X, addConst=True):
    """
    Breusch-Pagan test for heteroskedasticity.
    
    Parameters:
        y: dependent variable
        X: independent variables
        addConst: add constant term
    
    Returns:
        dict with testStatistic, pValue, df, conclusion
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
        'conclusion': conclusion
    }

def durbinWatson(residuals):
    """
    Durbin-Watson statistic for autocorrelation.
    DW ~ 2: no autocorrelation
    0 < DW < 2: positive autocorrelation
    2 < DW < 4: negative autocorrelation
    
    Parameters:
        residuals: residuals from regression
    
    Returns:
        float: Durbin-Watson statistic
    """
    residuals = np.asarray(residuals)
    
    diffSquared = np.sum(np.diff(residuals)**2)
    residSquared = np.sum(residuals**2)
    
    return diffSquared / residSquared

def ljungBox(residuals, lags=None):
    """
    Ljung-Box test for autocorrelation in residuals.
    
    Parameters:
        residuals: residuals from regression
        lags: number of lags (default min(10, n//5))
    
    Returns:
        dict with lags, autocorrelations, qStats, pValues, criticalValues
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
        'criticalValues': critVals
    }

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

def adfTest(series, lags=None, regression='c', autolag='AIC'):
    """
    Augmented Dickey-Fuller test for unit root.
    
    Parameters:
        series: time series
        lags: number of lags (auto if None)
        regression: 'nc' (no constant), 'c' (constant), 'ct' (constant+trend)
        autolag: 'AIC' or 'BIC'
    
    Returns:
        dict with testStatistic, pValue, criticalValues, conclusion, optimalLag
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
        'optimalLag': optimalLag
    }

def kpssTest(series, lags=None, regression='c'):
    """
    KPSS test for stationarity.
    
    Parameters:
        series: time series
        lags: number of lags (auto if None)
        regression: 'c' (constant) or 'ct' (constant+trend)
    
    Returns:
        dict with testStatistic, pValue, criticalValues, conclusion, lags
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
        'lags': lags
    }

def grangerCausality(y, x, maxLag=4):
    """
    Granger causality test: does x Granger-cause y?
    
    Parameters:
        y: dependent variable (time series)
        x: independent variable (time series)
        maxLag: maximum lag to test
    
    Returns:
        dict with lags, fStats, pValues, conclusion
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
        'conclusion': conclusion
    }