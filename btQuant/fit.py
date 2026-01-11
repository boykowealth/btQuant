import numpy as np

def fitGbm(prices, dt=1/252):
    """
    Fit Geometric Brownian Motion to price data.
    
    Parameters:
        prices: array of prices
        dt: time step (default 1/252 for daily)
    
    Returns:
        dict with mu (drift), sigma (volatility)
    """
    prices = np.asarray(prices, dtype=float)
    logReturns = np.diff(np.log(prices))
    
    variance = np.var(logReturns, ddof=1)
    mu = np.mean(logReturns) / dt + 0.5 * variance / dt
    sigma = np.sqrt(variance / dt)
    
    return {'mu': mu, 'sigma': sigma}

def fitOu(spread, dt=1/252):
    """
    Fit Ornstein-Uhlenbeck process to spread data.
    
    Parameters:
        spread: array of spread/price data
        dt: time step
    
    Returns:
        dict with theta (mean reversion), mu (long-term mean), sigma, halfLife
    """
    spread = np.asarray(spread, dtype=float)
    n = len(spread) - 1
    
    x = spread[:-1]
    y = spread[1:]
    
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x * x)
    Syy = np.sum(y * y)
    Sxy = np.sum(x * y)
    
    denom = n * (Sxx - Sxy) - (Sx * Sx - Sx * Sy)
    mu = (Sy * Sxx - Sx * Sxy) / denom
    
    numerator = Sxy - mu * Sx - mu * Sy + n * mu * mu
    denominator = Sxx - 2 * mu * Sx + n * mu * mu
    theta = -np.log(numerator / denominator)
    
    a = np.exp(-theta)
    sigmah2 = (Syy - 2 * a * Sxy + a * a * Sxx - 
               2 * mu * (1 - a) * (Sy - a * Sx) + 
               n * mu * mu * (1 - a) * (1 - a)) / n
    sigma = np.sqrt(sigmah2 * 2 * theta / (1 - a * a))
    
    halfLife = np.log(2) / theta / dt
    
    return {
        'theta': theta / dt,
        'mu': mu,
        'sigma': sigma * np.sqrt(1 / dt),
        'halfLife': halfLife
    }

def fitLevyOu(spread, jumpDetectionThreshold=0.4, dt=1/252):
    """
    Fit Lévy OU (jump-diffusion OU) model to spread data.
    
    Parameters:
        spread: array of spread/price data
        jumpDetectionThreshold: Bayesian jump detection threshold
        dt: time step
    
    Returns:
        dict with theta, mu, sigma, halfLife, jumpLambda, jumpMu, jumpSigma
    """
    spread = np.asarray(spread, dtype=float)
    
    diffs = np.diff(spread)
    n = len(diffs)
    
    mad = np.median(np.abs(diffs - np.median(diffs)))
    sigmaEst = 1.4826 * mad
    
    def normPdf(x, mean, std):
        return np.exp(-0.5 * ((x - mean) / std)**2) / (std * np.sqrt(2 * np.pi))
    
    pPrior = 0.01
    pNoJump = normPdf(diffs, 0, sigmaEst)
    pJump = normPdf(diffs, diffs, sigmaEst)
    
    likelihoodRatio = (pPrior * pJump) / ((1 - pPrior) * pNoJump + pPrior * pJump)
    
    jumpMask = likelihoodRatio > jumpDetectionThreshold
    jumpIndices = np.where(jumpMask)[0] + 1
    jumpSizes = diffs[jumpMask]
    
    mask = np.ones(len(spread), dtype=bool)
    if len(jumpIndices) > 0:
        mask[jumpIndices] = False
        afterIndices = jumpIndices + 1
        afterIndices = afterIndices[afterIndices < len(mask)]
        if len(afterIndices) > 0:
            mask[afterIndices] = False
    
    cleanSpread = spread[mask]
    
    if len(cleanSpread) < 10:
        cleanSpread = spread
    
    unconditionalStd = np.std(cleanSpread, ddof=1)
    
    if len(cleanSpread) > 1:
        lag1Corr = np.corrcoef(cleanSpread[:-1], cleanSpread[1:])[0, 1]
        lag1Corr = np.clip(lag1Corr, 0.01, 0.99)
        thetaEst = -np.log(lag1Corr) / dt
        thetaEst = np.clip(thetaEst, 0.1, 1 / dt)
    else:
        thetaEst = 1.0
    
    sigmaEst = unconditionalStd * np.sqrt(2 * thetaEst)
    muEst = np.mean(cleanSpread)
    
    def negLogLikelihood(params):
        theta, mu, sigma = params
        X = cleanSpread[:-1]
        Y = cleanSpread[1:]
        
        drift = X + (mu - X) * (1 - np.exp(-theta * dt))
        variance = np.maximum(sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta), 1e-10)
        
        return -np.sum(-0.5 * np.log(2 * np.pi * variance) - (Y - drift)**2 / (2 * variance))
    
    params = np.array([thetaEst, muEst, sigmaEst])
    
    for iteration in range(50):
        ll = negLogLikelihood(params)
        
        eps = 1e-8
        grad = np.zeros(3)
        for i in range(3):
            paramsPlus = params.copy()
            paramsPlus[i] += eps
            grad[i] = (negLogLikelihood(paramsPlus) - ll) / eps
        
        params -= 0.001 * grad
        
        params[0] = np.clip(params[0], 1e-4, 1 / dt)
        params[2] = np.clip(params[2], 1e-4, 2.0 * sigmaEst)
    
    theta, mu, sigma = params
    halfLife = np.log(2) / theta / dt
    
    jumpLambda = len(jumpIndices) / len(spread)
    jumpMu = np.mean(jumpSizes) if len(jumpSizes) > 0 else 0.0
    jumpSigma = np.std(jumpSizes, ddof=1) if len(jumpSizes) > 1 else 0.0
    
    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'halfLife': halfLife,
        'jumpLambda': jumpLambda,
        'jumpMu': jumpMu,
        'jumpSigma': jumpSigma
    }

def fitAr1(series):
    """
    Fit AR(1) model.
    
    Parameters:
        series: time series array
    
    Returns:
        dict with phi (AR coefficient), intercept, sigma2 (error variance)
    """
    series = np.asarray(series, dtype=float)
    
    meanSeries = np.mean(series)
    seriesDemeaned = series - meanSeries
    
    phi = np.dot(seriesDemeaned[:-1], seriesDemeaned[1:]) / np.dot(seriesDemeaned[:-1], seriesDemeaned[:-1])
    
    residuals = seriesDemeaned[1:] - phi * seriesDemeaned[:-1]
    sigma2 = np.var(residuals, ddof=1)
    
    intercept = meanSeries * (1 - phi)
    
    return {'phi': phi, 'intercept': intercept, 'sigma2': sigma2}

def fitArma(series, p=1, q=1, maxIter=100):
    """
    Fit ARMA(p,q) model using approximate method.
    
    Parameters:
        series: time series
        p: AR order
        q: MA order
        maxIter: maximum iterations
    
    Returns:
        dict with arCoefs, maCoefs, sigma2
    """
    series = np.asarray(series, dtype=float)
    series = series - np.mean(series)
    
    n = len(series)
    
    arCoefs = np.zeros(p)
    for i in range(p):
        if i < n - 1:
            arCoefs[i] = np.corrcoef(series[:-i-1], series[i+1:])[0, 1] * 0.5
    
    maCoefs = np.zeros(q)
    
    residuals = series.copy()
    for iteration in range(min(maxIter, 10)):
        for i in range(p):
            if i < n - 1:
                residuals[i+1:] -= arCoefs[i] * series[:n-i-1]
    
    sigma2 = np.var(residuals, ddof=1)
    
    return {'arCoefs': arCoefs, 'maCoefs': maCoefs, 'sigma2': sigma2}

def fitGarch(series, p=1, q=1, maxIter=50):
    """
    Fit GARCH(p,q) model.
    
    Parameters:
        series: returns series
        p: ARCH order
        q: GARCH order
        maxIter: maximum iterations
    
    Returns:
        dict with omega, alphas, betas, aic, bic
    """
    series = np.asarray(series, dtype=float)
    n = len(series)
    
    omega = np.var(series, ddof=1) * 0.1
    alphas = np.ones(p) * 0.1 / p
    betas = np.ones(q) * 0.8 / q
    
    params = np.concatenate([[omega], alphas, betas])
    
    def negLogLikelihood(params):
        omega = params[0]
        alphas = params[1:p+1]
        betas = params[p+1:]
        
        sigma2 = np.zeros(n)
        sigma2[:max(p, q)] = np.var(series, ddof=1)
        
        epsilon2 = series * series
        
        for t in range(max(p, q), n):
            sigma2[t] = omega
            for i in range(p):
                if t - i - 1 >= 0:
                    sigma2[t] += alphas[i] * epsilon2[t - i - 1]
            for j in range(q):
                if t - j - 1 >= 0:
                    sigma2[t] += betas[j] * sigma2[t - j - 1]
        
        sigma2 = np.maximum(sigma2, 1e-6)
        
        logLik = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + epsilon2 / sigma2)
        return -logLik
    
    for iteration in range(maxIter):
        ll = negLogLikelihood(params)
        
        eps = 1e-8
        grad = np.zeros(len(params))
        for i in range(len(params)):
            paramsPlus = params.copy()
            paramsPlus[i] += eps
            grad[i] = (negLogLikelihood(paramsPlus) - ll) / eps
        
        params -= 0.001 * grad
        params[0] = max(params[0], 1e-6)
        params[1:] = np.clip(params[1:], 0, 0.99)
    
    ll = -negLogLikelihood(params)
    k = len(params)
    aic = 2 * k - 2 * ll
    bic = np.log(n) * k - 2 * ll
    
    omega = params[0]
    alphas = params[1:p+1]
    betas = params[p+1:]
    
    paramsDict = {'omega': omega}
    for i in range(p):
        paramsDict[f'alpha{i+1}'] = alphas[i]
    for j in range(q):
        paramsDict[f'beta{j+1}'] = betas[j]
    
    return {'params': paramsDict, 'aic': aic, 'bic': bic}

def fitHeston(prices, dt=1/252):
    """
    Fit Heston stochastic volatility model.
    
    Parameters:
        prices: price series
        dt: time step
    
    Returns:
        dict with mu, kappa, theta, sigmaV, rho, v0
    """
    prices = np.asarray(prices, dtype=float)
    logReturns = np.diff(np.log(prices))
    
    realizedVar = logReturns**2 / dt
    
    kappa = 2 * (1 - np.corrcoef(realizedVar[:-1], realizedVar[1:])[0, 1]) / dt
    theta = np.mean(realizedVar)
    
    sigmaV = np.std(np.diff(realizedVar), ddof=1) * np.sqrt(1 / dt)
    
    rho = np.corrcoef(logReturns[:-1], np.diff(realizedVar))[0, 1]
    
    mu = np.mean(logReturns) / dt
    
    return {
        'mu': mu,
        'kappa': kappa,
        'theta': theta,
        'sigmaV': sigmaV,
        'rho': rho,
        'v0': realizedVar[0]
    }

def fitCir(rates, dt=1/252):
    """
    Fit Cox-Ingersoll-Ross model to interest rate data.
    
    Parameters:
        rates: interest rate series
        dt: time step
    
    Returns:
        dict with kappa, theta, sigma
    """
    rates = np.asarray(rates, dtype=float)
    
    rates = np.maximum(rates, 1e-6)
    
    x = rates[:-1]
    y = rates[1:]
    
    n = len(x)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x * x)
    Sxy = np.sum(x * y)
    
    denom = n * Sxx - Sx * Sx
    a = (Sy * Sxx - Sx * Sxy) / denom
    b = (n * Sxy - Sx * Sy) / denom
    
    theta = a / (1 - b)
    kappa = -np.log(b) / dt
    
    residuals = y - a - b * x
    sqrtX = np.sqrt(x)
    sigma = np.std(residuals / sqrtX, ddof=1) * np.sqrt(1 / dt)
    
    return {'kappa': kappa, 'theta': theta, 'sigma': sigma}

def fitVasicek(rates, dt=1/252):
    """
    Fit Vasicek model to interest rate data.
    
    Parameters:
        rates: interest rate series
        dt: time step
    
    Returns:
        dict with kappa, theta, sigma
    """
    rates = np.asarray(rates, dtype=float)
    
    x = rates[:-1]
    y = rates[1:]
    
    n = len(x)
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxx = np.sum(x * x)
    Sxy = np.sum(x * y)
    
    denom = n * Sxx - Sx * Sx
    a = (Sy * Sxx - Sx * Sxy) / denom
    b = (n * Sxy - Sx * Sy) / denom
    
    theta = a / (1 - b)
    kappa = -np.log(b) / dt
    
    residuals = y - a - b * x
    sigma = np.std(residuals, ddof=1) * np.sqrt(1 / dt)
    
    return {'kappa': kappa, 'theta': theta, 'sigma': sigma}

def fitJumpDiffusion(prices, dt=1/252, threshold=3.0):
    """
    Fit jump-diffusion model by separating jumps from diffusion.
    
    Parameters:
        prices: price series
        dt: time step
        threshold: jump detection threshold (std devs)
    
    Returns:
        dict with mu, sigma, jumpLambda, jumpMu, jumpSigma
    """
    prices = np.asarray(prices, dtype=float)
    logReturns = np.diff(np.log(prices))
    
    medianReturn = np.median(logReturns)
    mad = np.median(np.abs(logReturns - medianReturn))
    robustStd = 1.4826 * mad
    
    jumpMask = np.abs(logReturns - medianReturn) > threshold * robustStd
    
    normalReturns = logReturns[~jumpMask]
    jumpReturns = logReturns[jumpMask]
    
    if len(normalReturns) > 0:
        mu = np.mean(normalReturns) / dt
        sigma = np.std(normalReturns, ddof=1) / np.sqrt(dt)
    else:
        mu = 0.0
        sigma = 0.1
    
    jumpLambda = len(jumpReturns) / len(logReturns) / dt
    
    if len(jumpReturns) > 0:
        jumpMu = np.mean(jumpReturns)
        jumpSigma = np.std(jumpReturns, ddof=1)
    else:
        jumpMu = 0.0
        jumpSigma = 0.0
    
    return {
        'mu': mu,
        'sigma': sigma,
        'jumpLambda': jumpLambda,
        'jumpMu': jumpMu,
        'jumpSigma': jumpSigma
    }

def fitCopula(data1, data2, copulaType='gaussian'):
    """
    Fit copula to bivariate data.
    
    Parameters:
        data1: first variable
        data2: second variable
        copulaType: 'gaussian' (only type supported)
    
    Returns:
        dict with rho (correlation parameter)
    """
    data1 = np.asarray(data1, dtype=float)
    data2 = np.asarray(data2, dtype=float)
    
    u1 = (np.argsort(np.argsort(data1)) + 1) / (len(data1) + 1)
    u2 = (np.argsort(np.argsort(data2)) + 1) / (len(data2) + 1)
    
    def normPpf(p):
        if p <= 0 or p >= 1:
            return 0
        if p < 0.5:
            sign = -1
            p = 1 - p
        else:
            sign = 1
        t = np.sqrt(-2 * np.log(1 - p))
        x = t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (1 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3)
        return sign * x
    
    z1 = np.array([normPpf(u) for u in u1])
    z2 = np.array([normPpf(u) for u in u2])
    
    rho = np.corrcoef(z1, z2)[0, 1]
    
    return {'rho': rho}

def fitDistributions(data, distributions=None):
    """
    Fit multiple distributions and rank by AIC.
    
    Parameters:
        data: array of observations
        distributions: list of distribution names (None = all common)
    
    Returns:
        list of (distName, params, aic) sorted by AIC
    """
    data = np.asarray(data, dtype=float)
    
    if distributions is None:
        distributions = ['normal', 'lognormal', 'exponential', 'gamma']
    
    results = []
    
    for distName in distributions:
        try:
            if distName == 'normal':
                mu = np.mean(data)
                sigma = np.std(data, ddof=1)
                logLik = -0.5 * len(data) * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)
                aic = 2 * 2 - 2 * logLik
                results.append((distName, {'mu': mu, 'sigma': sigma}, aic))
            
            elif distName == 'lognormal':
                positiveData = data[data > 0]
                if len(positiveData) > 0:
                    logData = np.log(positiveData)
                    mu = np.mean(logData)
                    sigma = np.std(logData, ddof=1)
                    logLik = -0.5 * len(positiveData) * np.log(2 * np.pi * sigma**2) - np.sum((logData - mu)**2) / (2 * sigma**2) - np.sum(np.log(positiveData))
                    aic = 2 * 2 - 2 * logLik
                    results.append((distName, {'mu': mu, 'sigma': sigma}, aic))
            
            elif distName == 'exponential':
                positiveData = data[data > 0]
                if len(positiveData) > 0:
                    lambdaParam = 1.0 / np.mean(positiveData)
                    logLik = len(positiveData) * np.log(lambdaParam) - lambdaParam * np.sum(positiveData)
                    aic = 2 * 1 - 2 * logLik
                    results.append((distName, {'lambda': lambdaParam}, aic))
            
            elif distName == 'gamma':
                positiveData = data[data > 0]
                if len(positiveData) > 0:
                    meanData = np.mean(positiveData)
                    varData = np.var(positiveData, ddof=1)
                    alpha = meanData**2 / varData
                    beta = meanData / varData
                    results.append((distName, {'alpha': alpha, 'beta': beta}, 0))
        
        except Exception:
            pass
    
    results.sort(key=lambda x: x[2])
    
    return results

def aic(logLikelihood, nParams):
    """
    Akaike Information Criterion.
    
    Parameters:
        logLikelihood: log-likelihood value
        nParams: number of parameters
    
    Returns:
        AIC value
    """
    return 2 * nParams - 2 * logLikelihood

def bic(logLikelihood, nParams, nObs):
    """
    Bayesian Information Criterion.
    
    Parameters:
        logLikelihood: log-likelihood value
        nParams: number of parameters
        nObs: number of observations
    
    Returns:
        BIC value
    """
    return np.log(nObs) * nParams - 2 * logLikelihood