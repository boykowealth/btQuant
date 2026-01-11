import numpy as np

def _normCdf(x):
    """Standard normal CDF approximation."""
    return 0.5 * (1.0 + np.tanh(x / np.sqrt(2.0) * 0.7978845608))

def _normPpf(p):
    """Standard normal percent point function (inverse CDF)."""
    if p <= 0 or p >= 1:
        return np.nan
    
    if p < 0.5:
        sign = -1
        p = 1 - p
    else:
        sign = 1
    
    t = np.sqrt(-2 * np.log(1 - p))
    
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308
    
    x = t - (c0 + c1 * t + c2 * t**2) / (1 + d1 * t + d2 * t**2 + d3 * t**3)
    
    return sign * x

def _normPdf(x):
    """Standard normal PDF."""
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

def parametricVar(returns, confidence=0.95):
    """
    Parametric Value at Risk (assumes normal distribution).
    
    Parameters:
        returns: array of returns
        confidence: confidence level (default 0.95)
    
    Returns:
        VaR estimate
    """
    mean = np.mean(returns)
    std = np.std(returns)
    
    zScore = _normPpf(1 - confidence)
    
    var = -(mean + zScore * std)
    return var

def historicalVar(returns, confidence=0.95):
    """
    Historical simulation Value at Risk.
    
    Parameters:
        returns: array of returns
        confidence: confidence level
    
    Returns:
        VaR estimate
    """
    sortedReturns = np.sort(returns)
    index = int((1 - confidence) * len(sortedReturns))
    var = -sortedReturns[index]
    return var

def parametricCvar(returns, confidence=0.95):
    """
    Parametric Conditional Value at Risk (assumes normal).
    
    Parameters:
        returns: array of returns
        confidence: confidence level
    
    Returns:
        CVaR estimate
    """
    mean = np.mean(returns)
    std = np.std(returns)
    
    zScore = _normPpf(1 - confidence)
    
    cvar = -(mean + std * (_normPdf(zScore) / (1 - confidence)))
    return cvar

def historicalCvar(returns, confidence=0.95):
    """
    Historical simulation Conditional Value at Risk.
    
    Parameters:
        returns: array of returns
        confidence: confidence level
    
    Returns:
        CVaR estimate
    """
    sortedReturns = np.sort(returns)
    index = int((1 - confidence) * len(sortedReturns))
    tailLosses = sortedReturns[:index + 1]
    cvar = -np.mean(tailLosses)
    return cvar

def expectedShortfall(returns, confidence=0.95):
    """
    Expected shortfall (ES), same as historical CVaR.
    
    Parameters:
        returns: array of returns
        confidence: confidence level
    
    Returns:
        ES estimate
    """
    return historicalCvar(returns, confidence)

def drawdown(returns):
    """
    Maximum drawdown.
    
    Parameters:
        returns: array of returns
    
    Returns:
        maximum drawdown (negative value)
    """
    cumReturns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumReturns)
    drawdowns = (cumReturns - peak) / (peak + 1)
    maxDrawdown = np.min(drawdowns)
    return maxDrawdown

def calmarRatio(returns, riskFreeRate=0):
    """
    Calmar ratio: annualized return / max drawdown.
    
    Parameters:
        returns: array of returns
        riskFreeRate: risk-free rate
    
    Returns:
        Calmar ratio
    """
    annualizedReturn = np.mean(returns) * 252
    maxDrawdown = drawdown(returns)
    calmar = annualizedReturn / abs(maxDrawdown) if maxDrawdown != 0 else np.inf
    return calmar

def sharpeRatio(returns, riskFreeRate=0):
    """
    Sharpe ratio.
    
    Parameters:
        returns: array of returns
        riskFreeRate: risk-free rate per period
    
    Returns:
        Sharpe ratio
    """
    excessReturns = returns - riskFreeRate
    return np.mean(excessReturns) / (np.std(excessReturns) + 1e-10)

def sortinoRatio(returns, riskFreeRate=0, target=0):
    """
    Sortino ratio (uses downside deviation).
    
    Parameters:
        returns: array of returns
        riskFreeRate: risk-free rate
        target: target return (default 0)
    
    Returns:
        Sortino ratio
    """
    excessReturns = returns - riskFreeRate
    downsideReturns = excessReturns[excessReturns < target]
    
    if len(downsideReturns) == 0:
        return np.inf
    
    downsideDev = np.sqrt(np.mean(downsideReturns**2))
    
    return np.mean(excessReturns) / (downsideDev + 1e-10)

def omegaRatio(returns, threshold=0.0):
    """
    Omega ratio: ratio of gains to losses.
    
    Parameters:
        returns: array of returns
        threshold: threshold return
    
    Returns:
        Omega ratio
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    
    sumGains = np.sum(gains) if len(gains) > 0 else 0
    sumLosses = np.sum(losses) if len(losses) > 0 else 1e-10
    
    omega = sumGains / sumLosses
    return omega

def modifiedVar(returns, confidence=0.95):
    """
    Modified VaR using Cornish-Fisher expansion for skewness and kurtosis.
    
    Parameters:
        returns: array of returns
        confidence: confidence level
    
    Returns:
        modified VaR
    """
    z = _normPpf(confidence)
    
    s = _skew(returns)
    k = _kurtosis(returns)
    
    zCf = z + (1 / 6) * (z**2 - 1) * s + (1 / 24) * (z**3 - 3 * z) * (k - 3) - (1 / 36) * (2 * z**3 - 5 * z) * s**2
    
    return -np.mean(returns) + zCf * np.std(returns)

def _skew(x):
    """Sample skewness."""
    n = len(x)
    mean = np.mean(x)
    m2 = np.sum((x - mean)**2) / n
    m3 = np.sum((x - mean)**3) / n
    return m3 / (m2**1.5 + 1e-10)

def _kurtosis(x):
    """Sample excess kurtosis."""
    n = len(x)
    mean = np.mean(x)
    m2 = np.sum((x - mean)**2) / n
    m4 = np.sum((x - mean)**4) / n
    return m4 / (m2**2 + 1e-10)

def hillTailIndex(returns, k=50):
    """
    Hill estimator for tail index (heavy tails).
    
    Parameters:
        returns: array of returns
        k: number of extreme values
    
    Returns:
        tail index (lower = heavier tail)
    """
    sortedReturns = -np.sort(-returns)
    topK = sortedReturns[:k]
    
    if len(topK) < 2:
        return np.nan
    
    return 1 / np.mean(np.log(topK / sortedReturns[k]))

def excessKurtosis(returns):
    """
    Excess kurtosis.
    
    Parameters:
        returns: array of returns
    
    Returns:
        excess kurtosis
    """
    return _kurtosis(returns)

def beta(assetReturns, marketReturns):
    """
    Beta coefficient (systematic risk).
    
    Parameters:
        assetReturns: asset returns
        marketReturns: market returns
    
    Returns:
        beta
    """
    covariance = np.cov(assetReturns, marketReturns)[0, 1]
    marketVariance = np.var(marketReturns)
    return covariance / (marketVariance + 1e-10)

def treynorRatio(returns, marketReturns, riskFreeRate=0):
    """
    Treynor ratio: excess return / beta.
    
    Parameters:
        returns: portfolio returns
        marketReturns: market returns
        riskFreeRate: risk-free rate
    
    Returns:
        Treynor ratio
    """
    excessReturn = np.mean(returns) - riskFreeRate
    portfolioBeta = beta(returns, marketReturns)
    return excessReturn / (portfolioBeta + 1e-10)

def informationRatio(returns, benchmarkReturns):
    """
    Information ratio: active return / tracking error.
    
    Parameters:
        returns: portfolio returns
        benchmarkReturns: benchmark returns
    
    Returns:
        information ratio
    """
    activeReturns = returns - benchmarkReturns
    return np.mean(activeReturns) / (np.std(activeReturns) + 1e-10)

def trackingError(returns, benchmarkReturns):
    """
    Tracking error (standard deviation of active returns).
    
    Parameters:
        returns: portfolio returns
        benchmarkReturns: benchmark returns
    
    Returns:
        tracking error
    """
    activeReturns = returns - benchmarkReturns
    return np.std(activeReturns)

def maxDrawdownDuration(returns):
    """
    Maximum drawdown duration in periods.
    
    Parameters:
        returns: array of returns
    
    Returns:
        maximum drawdown duration
    """
    cumReturns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumReturns)
    
    underwater = cumReturns < peak
    
    maxDuration = 0
    currentDuration = 0
    
    for isUnderwater in underwater:
        if isUnderwater:
            currentDuration += 1
            maxDuration = max(maxDuration, currentDuration)
        else:
            currentDuration = 0
    
    return maxDuration

def valueAtRisk(returns, confidence=0.95, method='historical'):
    """
    Value at Risk with method selection.
    
    Parameters:
        returns: array of returns
        confidence: confidence level
        method: 'historical' or 'parametric'
    
    Returns:
        VaR estimate
    """
    if method == 'historical':
        return historicalVar(returns, confidence)
    elif method == 'parametric':
        return parametricVar(returns, confidence)
    else:
        raise ValueError("method must be 'historical' or 'parametric'")

def conditionalValueAtRisk(returns, confidence=0.95, method='historical'):
    """
    Conditional Value at Risk with method selection.
    
    Parameters:
        returns: array of returns
        confidence: confidence level
        method: 'historical' or 'parametric'
    
    Returns:
        CVaR estimate
    """
    if method == 'historical':
        return historicalCvar(returns, confidence)
    elif method == 'parametric':
        return parametricCvar(returns, confidence)
    else:
        raise ValueError("method must be 'historical' or 'parametric'")

def downsideDeviation(returns, target=0):
    """
    Downside deviation.
    
    Parameters:
        returns: array of returns
        target: target return
    
    Returns:
        downside deviation
    """
    downsideReturns = returns[returns < target] - target
    if len(downsideReturns) == 0:
        return 0.0
    return np.sqrt(np.mean(downsideReturns**2))

def ulcerIndex(returns):
    """
    Ulcer index (downside volatility measure).
    
    Parameters:
        returns: array of returns
    
    Returns:
        ulcer index
    """
    cumReturns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumReturns)
    drawdowns = (cumReturns - peak) / (peak + 1)
    return np.sqrt(np.mean(drawdowns**2))

def painIndex(returns):
    """
    Pain index (average squared drawdown).
    
    Parameters:
        returns: array of returns
    
    Returns:
        pain index
    """
    cumReturns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumReturns)
    drawdowns = (cumReturns - peak) / (peak + 1)
    return np.mean(drawdowns**2)

def tailRatio(returns, confidence=0.95):
    """
    Tail ratio: right tail / left tail.
    
    Parameters:
        returns: array of returns
        confidence: confidence level
    
    Returns:
        tail ratio (>1 means right tail heavier)
    """
    rightTail = np.percentile(returns, confidence * 100)
    leftTail = np.percentile(returns, (1 - confidence) * 100)
    return abs(rightTail / leftTail) if leftTail != 0 else np.inf

def capturRatio(returns, marketReturns):
    """
    Upside and downside capture ratios.
    
    Parameters:
        returns: portfolio returns
        marketReturns: market returns
    
    Returns:
        dict with upsideCapture, downsideCapture, captureRatio
    """
    upMarket = marketReturns > 0
    downMarket = marketReturns < 0
    
    if np.sum(upMarket) > 0:
        upsideCapture = np.mean(returns[upMarket]) / np.mean(marketReturns[upMarket])
    else:
        upsideCapture = 0.0
    
    if np.sum(downMarket) > 0:
        downsideCapture = np.mean(returns[downMarket]) / np.mean(marketReturns[downMarket])
    else:
        downsideCapture = 0.0
    
    captureRatio = upsideCapture / downsideCapture if downsideCapture != 0 else np.inf
    
    return {
        'upsideCapture': upsideCapture,
        'downsideCapture': downsideCapture,
        'captureRatio': captureRatio
    }

def stabilityRatio(returns):
    """
    Stability of returns (R-squared of linear regression).
    
    Parameters:
        returns: array of returns
    
    Returns:
        stability ratio (0-1, higher = more stable)
    """
    cumReturns = np.cumsum(returns)
    x = np.arange(len(cumReturns))
    
    xMean = np.mean(x)
    yMean = np.mean(cumReturns)
    
    numerator = np.sum((x - xMean) * (cumReturns - yMean))
    denominator = np.sqrt(np.sum((x - xMean)**2) * np.sum((cumReturns - yMean)**2))
    
    if denominator == 0:
        return 0.0
    
    correlation = numerator / denominator
    return correlation**2