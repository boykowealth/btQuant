import numpy as np

def _normCdf(x):
    """Standard normal CDF."""
    return 0.5 * (1.0 + np.tanh(x / np.sqrt(2.0) * 0.7978845608))

def _normPdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def _normPpf(p):
    """Standard normal quantile function."""
    if p <= 0 or p >= 1:
        return np.nan
    
    if p < 0.5:
        sign = -1
        p = 1 - p
    else:
        sign = 1
    
    t = np.sqrt(-2 * np.log(1 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    x = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
    
    return sign * x

def _chiSqCdf(x, df):
    """Chi-squared CDF approximation."""
    if x <= 0:
        return 0.0
    if df == 1:
        return 2 * _normCdf(np.sqrt(x)) - 1
    return _gammaInc(df / 2, x / 2)

def _gammaInc(a, x):
    """Incomplete gamma function."""
    if x < 0 or a <= 0:
        return 0.0
    
    if x < a + 1:
        ap, delta, sumVal = a, 1.0 / a, 1.0 / a
        for n in range(1, 100):
            ap += 1
            delta *= x / ap
            sumVal += delta
            if delta < sumVal * 1e-10:
                break
        return sumVal * np.exp(-x + a * np.log(x) - _logGamma(a))
    else:
        b, c, d, h = x + 1 - a, 1.0 / 1e-30, 1.0 / b, 1.0 / b
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

def _logGamma(x):
    """Log gamma function."""
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

def fitNormal(data):
    """
    Fit normal distribution.
    
    Parameters:
        data: array of observations
    
    Returns:
        dict with mu, sigma, logLikelihood
    """
    data = np.asarray(data)
    mu = np.mean(data)
    sigma = np.std(data, ddof=1)
    
    logLik = -0.5 * len(data) * np.log(2 * np.pi * sigma**2) - np.sum((data - mu)**2) / (2 * sigma**2)
    
    return {'mu': mu, 'sigma': sigma, 'logLikelihood': logLik}

def fitLognormal(data):
    """
    Fit lognormal distribution.
    
    Parameters:
        data: array of positive observations
    
    Returns:
        dict with mu, sigma (of log), logLikelihood
    """
    data = np.asarray(data)
    data = data[data > 0]
    
    logData = np.log(data)
    mu = np.mean(logData)
    sigma = np.std(logData, ddof=1)
    
    logLik = -0.5 * len(data) * np.log(2 * np.pi * sigma**2) - np.sum((logData - mu)**2) / (2 * sigma**2)
    logLik -= np.sum(np.log(data))
    
    return {'mu': mu, 'sigma': sigma, 'logLikelihood': logLik}

def fitExponential(data):
    """
    Fit exponential distribution.
    
    Parameters:
        data: array of positive observations
    
    Returns:
        dict with lambda (rate), logLikelihood
    """
    data = np.asarray(data)
    data = data[data > 0]
    
    lambdaParam = 1.0 / np.mean(data)
    
    logLik = len(data) * np.log(lambdaParam) - lambdaParam * np.sum(data)
    
    return {'lambda': lambdaParam, 'logLikelihood': logLik}

def fitGamma(data, maxIter=100):
    """
    Fit gamma distribution using method of moments.
    
    Parameters:
        data: array of positive observations
        maxIter: maximum iterations
    
    Returns:
        dict with alpha (shape), beta (rate), logLikelihood
    """
    data = np.asarray(data)
    data = data[data > 0]
    
    meanData = np.mean(data)
    varData = np.var(data, ddof=1)
    
    alpha = meanData**2 / varData
    beta = meanData / varData
    
    logLik = len(data) * (alpha * np.log(beta) - _logGamma(alpha)) + (alpha - 1) * np.sum(np.log(data)) - beta * np.sum(data)
    
    return {'alpha': alpha, 'beta': beta, 'logLikelihood': logLik}

def fitBeta(data, maxIter=100):
    """
    Fit beta distribution using method of moments.
    
    Parameters:
        data: array of observations in (0, 1)
        maxIter: maximum iterations
    
    Returns:
        dict with alpha, beta parameters, logLikelihood
    """
    data = np.asarray(data)
    data = data[(data > 0) & (data < 1)]
    
    meanData = np.mean(data)
    varData = np.var(data, ddof=1)
    
    commonTerm = meanData * (1 - meanData) / varData - 1
    alpha = meanData * commonTerm
    beta = (1 - meanData) * commonTerm
    
    alpha = max(alpha, 0.1)
    beta = max(beta, 0.1)
    
    logLik = len(data) * (_logGamma(alpha + beta) - _logGamma(alpha) - _logGamma(beta))
    logLik += (alpha - 1) * np.sum(np.log(data)) + (beta - 1) * np.sum(np.log(1 - data))
    
    return {'alpha': alpha, 'beta': beta, 'logLikelihood': logLik}

def fitT(data, maxIter=50):
    """
    Fit Student's t distribution.
    
    Parameters:
        data: array of observations
        maxIter: maximum iterations
    
    Returns:
        dict with df (degrees of freedom), mu, sigma, logLikelihood
    """
    data = np.asarray(data)
    
    mu = np.median(data)
    sigma = np.std(data, ddof=1)
    
    kurt = _kurtosis(data)
    if kurt > 0:
        df = 6.0 / kurt + 4
    else:
        df = 10.0
    
    df = max(df, 2.5)
    
    logLik = 0.0
    
    return {'df': df, 'mu': mu, 'sigma': sigma, 'logLikelihood': logLik}

def _kurtosis(x):
    """Excess kurtosis."""
    n = len(x)
    mean = np.mean(x)
    m2 = np.sum((x - mean)**2) / n
    m4 = np.sum((x - mean)**4) / n
    return m4 / (m2**2 + 1e-10) - 3

def _skewness(x):
    """Skewness."""
    n = len(x)
    mean = np.mean(x)
    m2 = np.sum((x - mean)**2) / n
    m3 = np.sum((x - mean)**3) / n
    return m3 / (m2**1.5 + 1e-10)

def moments(data):
    """
    Calculate distribution moments.
    
    Parameters:
        data: array of observations
    
    Returns:
        dict with mean, variance, skewness, kurtosis
    """
    data = np.asarray(data)
    
    return {
        'mean': np.mean(data),
        'variance': np.var(data, ddof=1),
        'skewness': _skewness(data),
        'kurtosis': _kurtosis(data)
    }

def ksTest(data, distName='normal', params=None):
    """
    Kolmogorov-Smirnov test for distribution fit.
    
    Parameters:
        data: array of observations
        distName: 'normal', 'lognormal', 'exponential'
        params: dict of distribution parameters (if None, estimated)
    
    Returns:
        dict with statistic, pValue (approximate)
    """
    data = np.asarray(data)
    n = len(data)
    sortedData = np.sort(data)
    
    if params is None:
        if distName == 'normal':
            params = fitNormal(data)
        elif distName == 'lognormal':
            params = fitLognormal(data)
        elif distName == 'exponential':
            params = fitExponential(data)
        else:
            raise ValueError("Unknown distribution")
    
    empiricalCdf = np.arange(1, n + 1) / n
    
    if distName == 'normal':
        theoreticalCdf = _normCdf((sortedData - params['mu']) / params['sigma'])
    elif distName == 'lognormal':
        theoreticalCdf = _normCdf((np.log(sortedData) - params['mu']) / params['sigma'])
    elif distName == 'exponential':
        theoreticalCdf = 1 - np.exp(-params['lambda'] * sortedData)
    else:
        raise ValueError("Unknown distribution")
    
    dPlus = np.max(empiricalCdf - theoreticalCdf)
    dMinus = np.max(theoreticalCdf - (empiricalCdf - 1 / n))
    
    statistic = max(dPlus, dMinus)
    
    pValue = np.exp(-2 * n * statistic**2)
    
    return {'statistic': statistic, 'pValue': pValue}

def adTest(data, distName='normal'):
    """
    Anderson-Darling test for normality.
    
    Parameters:
        data: array of observations
        distName: currently only 'normal' supported
    
    Returns:
        dict with statistic, critical values
    """
    data = np.asarray(data)
    n = len(data)
    
    if distName != 'normal':
        raise ValueError("Only normal distribution currently supported")
    
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    
    z = (data - mean) / std
    z = np.sort(z)
    
    i = np.arange(1, n + 1)
    cdf = _normCdf(z)
    
    S = -n - np.sum((2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1]))) / n
    
    criticalValues = {
        '15%': 0.576,
        '10%': 0.656,
        '5%': 0.787,
        '2.5%': 0.918,
        '1%': 1.092
    }
    
    return {'statistic': S, 'criticalValues': criticalValues}

def klDivergence(p, q):
    """
    Kullback-Leibler divergence.
    
    Parameters:
        p: true distribution (probabilities)
        q: approximate distribution (probabilities)
    
    Returns:
        KL divergence
    """
    p = np.asarray(p)
    q = np.asarray(q)
    
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    p = np.clip(p, 1e-12, 1)
    q = np.clip(q, 1e-12, 1)
    
    return np.sum(p * np.log(p / q))

def jsDivergence(p, q):
    """
    Jensen-Shannon divergence.
    
    Parameters:
        p: first distribution
        q: second distribution
    
    Returns:
        JS divergence
    """
    p = np.asarray(p)
    q = np.asarray(q)
    
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    m = 0.5 * (p + q)
    
    return 0.5 * (klDivergence(p, m) + klDivergence(q, m))

def fitMixture(data, nComponents=2, maxIter=100):
    """
    Fit Gaussian mixture model using EM algorithm.
    
    Parameters:
        data: array of observations
        nComponents: number of mixture components
        maxIter: maximum iterations
    
    Returns:
        dict with means, sigmas, weights
    """
    data = np.asarray(data).reshape(-1, 1)
    n = len(data)
    
    means = np.linspace(np.min(data), np.max(data), nComponents).reshape(-1, 1)
    sigmas = np.ones((nComponents, 1)) * np.std(data)
    weights = np.ones(nComponents) / nComponents
    
    for iteration in range(maxIter):
        responsibilities = np.zeros((n, nComponents))
        
        for k in range(nComponents):
            diff = data - means[k]
            responsibilities[:, k] = (weights[k] * 
                                     np.exp(-0.5 * (diff / sigmas[k])**2).flatten() / 
                                     (sigmas[k] * np.sqrt(2 * np.pi)))
        
        responsibilities /= (np.sum(responsibilities, axis=1, keepdims=True) + 1e-10)
        
        nK = np.sum(responsibilities, axis=0)
        
        for k in range(nComponents):
            means[k] = np.sum(responsibilities[:, k:k+1] * data, axis=0) / (nK[k] + 1e-10)
            diff = data - means[k]
            sigmas[k] = np.sqrt(np.sum(responsibilities[:, k:k+1] * diff**2, axis=0) / (nK[k] + 1e-10))
            weights[k] = nK[k] / n
    
    return {
        'means': means.flatten(),
        'sigmas': sigmas.flatten(),
        'weights': weights
    }

def quantile(data, q):
    """
    Calculate quantile.
    
    Parameters:
        data: array of observations
        q: quantile level (0 to 1)
    
    Returns:
        quantile value
    """
    data = np.asarray(data)
    return np.percentile(data, q * 100)

def qqPlot(data, distName='normal'):
    """
    Generate Q-Q plot data.
    
    Parameters:
        data: array of observations
        distName: reference distribution
    
    Returns:
        dict with theoretical and empirical quantiles
    """
    data = np.asarray(data)
    n = len(data)
    sortedData = np.sort(data)
    
    probabilities = (np.arange(1, n + 1) - 0.5) / n
    
    if distName == 'normal':
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        theoreticalQuantiles = mean + std * np.array([_normPpf(p) for p in probabilities])
    else:
        raise ValueError("Only normal distribution currently supported")
    
    return {
        'theoretical': theoreticalQuantiles,
        'empirical': sortedData
    }