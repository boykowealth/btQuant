import numpy as np

def _normCdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + np.tanh(x / np.sqrt(2.0) * 0.7978845608))

def _normPdf(x):
    """Standard normal probability density function."""
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)

def blackScholes(S, K, T, r, sigma, b=None, optType='call'):
    """
    European option pricing using Black-Scholes model with cost of carry.
    
    Parameters:
        S: current asset price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        b: cost of carry (default r)
        optType: 'call' or 'put'
    
    Returns:
        dict with price, delta, gamma, vega, rho, theta
    """
    if b is None:
        b = r
    
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    
    expBRT = np.exp((b - r) * T)
    expRT = np.exp(-r * T)
    
    if optType == 'call':
        price = S * expBRT * _normCdf(d1) - K * expRT * _normCdf(d2)
        delta = expBRT * _normCdf(d1)
        rho = K * T * expRT * _normCdf(d2)
        theta = (-S * expBRT * _normPdf(d1) * sigma / (2 * sqrtT) - 
                (b - r) * S * expBRT * _normCdf(d1) - 
                r * K * expRT * _normCdf(d2))
    else:
        price = K * expRT * _normCdf(-d2) - S * expBRT * _normCdf(-d1)
        delta = -expBRT * _normCdf(-d1)
        rho = -K * T * expRT * _normCdf(-d2)
        theta = (-S * expBRT * _normPdf(d1) * sigma / (2 * sqrtT) + 
                (b - r) * S * expBRT * _normCdf(-d1) + 
                r * K * expRT * _normCdf(-d2))
    
    gamma = expBRT * _normPdf(d1) / (S * sigma * sqrtT)
    vega = S * expBRT * _normPdf(d1) * sqrtT
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta
    }

def binomial(S, K, T, r, sigma, b=None, N=100, optType='call', american=False):
    """
    European or American option pricing using binomial tree.
    
    Parameters:
        S: current asset price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        b: cost of carry (default r)
        N: number of time steps
        optType: 'call' or 'put'
        american: American style if True
    
    Returns:
        dict with price, delta, gamma, theta
    """
    if b is None:
        b = r
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    q = (np.exp(b * dt) - d) / (u - d)
    disc = np.exp(-r * dt)
    
    prices = np.zeros((N + 1, N + 1))
    values = np.zeros((N + 1, N + 1))
    
    for i in range(N + 1):
        for j in range(i + 1):
            prices[j, i] = S * (u ** (i - j)) * (d ** j)
    
    if optType == 'call':
        values[:, N] = np.maximum(prices[:, N] - K, 0)
    else:
        values[:, N] = np.maximum(K - prices[:, N], 0)
    
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            values[j, i] = disc * (q * values[j, i + 1] + (1 - q) * values[j + 1, i + 1])
            if american:
                if optType == 'call':
                    values[j, i] = max(values[j, i], prices[j, i] - K)
                else:
                    values[j, i] = max(values[j, i], K - prices[j, i])
    
    price = values[0, 0]
    
    if N > 1:
        delta = (values[0, 1] - values[1, 1]) / (prices[0, 1] - prices[1, 1])
        if N > 2:
            gamma = ((values[0, 2] - values[1, 2]) / (prices[0, 2] - prices[1, 2]) - 
                    (values[1, 2] - values[2, 2]) / (prices[1, 2] - prices[2, 2])) / ((prices[0, 2] - prices[2, 2]) / 2)
        else:
            gamma = 0.0
    else:
        delta = 0.0
        gamma = 0.0
    
    theta = (values[0, 1] - price) / dt if N > 0 else 0.0
    
    return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta}

def trinomial(S, K, T, r, sigma, b=None, N=50, optType='call', american=False):
    """
    European or American option pricing using trinomial tree.
    
    Parameters:
        S: current asset price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        b: cost of carry (default r)
        N: number of time steps
        optType: 'call' or 'put'
        american: American style if True
    
    Returns:
        dict with price, delta, gamma, theta
    """
    if b is None:
        b = r
    
    dt = T / N
    dx = sigma * np.sqrt(3 * dt)
    
    pu = 1.0 / 6.0 + (b - 0.5 * sigma**2) * dt / (2 * dx)
    pd = 1.0 / 6.0 - (b - 0.5 * sigma**2) * dt / (2 * dx)
    pm = 2.0 / 3.0
    
    disc = np.exp(-r * dt)
    
    assetPrices = S * np.exp(np.arange(-N, N + 1) * dx)
    
    if optType == 'call':
        values = np.maximum(assetPrices - K, 0)
    else:
        values = np.maximum(K - assetPrices, 0)
    
    for i in range(N - 1, -1, -1):
        newValues = np.zeros(2 * i + 1)
        for j in range(2 * i + 1):
            idx = j - i + N
            newValues[j] = disc * (pu * values[idx + 2] + pm * values[idx + 1] + pd * values[idx])
            
            if american:
                currentPrice = S * np.exp((j - i) * dx)
                if optType == 'call':
                    newValues[j] = max(newValues[j], currentPrice - K)
                else:
                    newValues[j] = max(newValues[j], K - currentPrice)
        
        values = newValues
    
    price = values[0]
    
    h = 0.01 * S
    
    assetPricesUp = S * (1 + h / S) * np.exp(np.arange(-N, N + 1) * dx)
    assetPricesDown = S * (1 - h / S) * np.exp(np.arange(-N, N + 1) * dx)
    
    if optType == 'call':
        valuesUp = np.maximum(assetPricesUp - K, 0)
        valuesDown = np.maximum(assetPricesDown - K, 0)
    else:
        valuesUp = np.maximum(K - assetPricesUp, 0)
        valuesDown = np.maximum(K - assetPricesDown, 0)
    
    for i in range(N - 1, -1, -1):
        newValuesUp = np.zeros(2 * i + 1)
        newValuesDown = np.zeros(2 * i + 1)
        for j in range(2 * i + 1):
            idx = j - i + N
            newValuesUp[j] = disc * (pu * valuesUp[idx + 2] + pm * valuesUp[idx + 1] + pd * valuesUp[idx])
            newValuesDown[j] = disc * (pu * valuesDown[idx + 2] + pm * valuesDown[idx + 1] + pd * valuesDown[idx])
        valuesUp = newValuesUp
        valuesDown = newValuesDown
    
    priceUp = valuesUp[0]
    priceDown = valuesDown[0]
    
    delta = (priceUp - priceDown) / (2 * h)
    gamma = (priceUp - 2 * price + priceDown) / (h**2)
    
    theta = 0.0
    if T > dt:
        dtTheta = dt
        assetPricesTheta = S * np.exp(np.arange(-N, N + 1) * dx)
        if optType == 'call':
            valuesTheta = np.maximum(assetPricesTheta - K, 0)
        else:
            valuesTheta = np.maximum(K - assetPricesTheta, 0)
        
        for i in range(N - 2, -1, -1):
            newValuesTheta = np.zeros(2 * i + 1)
            for j in range(2 * i + 1):
                idx = j - i + N
                newValuesTheta[j] = disc * (pu * valuesTheta[idx + 2] + pm * valuesTheta[idx + 1] + pd * valuesTheta[idx])
            valuesTheta = newValuesTheta
        
        priceTheta = valuesTheta[0]
        theta = (priceTheta - price) / dtTheta
    
    return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta}

def asian(S, K, T, r, sigma, b=None, nSteps=100, optType='call'):
    """
    Asian option pricing with geometric averaging using analytic formula.
    
    Parameters:
        S: current asset price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        b: cost of carry (default r)
        nSteps: number of averaging steps
        optType: 'call' or 'put'
    
    Returns:
        dict with price, delta, gamma, vega, rho, theta
    """
    if b is None:
        b = r
    
    dt = T / nSteps
    
    sigmaAdj = sigma * np.sqrt((nSteps + 1) * (2 * nSteps + 1) / (6 * nSteps**2))
    bAdj = 0.5 * (b - 0.5 * sigma**2) + 0.5 * sigmaAdj**2
    
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (bAdj + 0.5 * sigmaAdj**2) * T) / (sigmaAdj * sqrtT)
    d2 = d1 - sigmaAdj * sqrtT
    
    expRT = np.exp(-r * T)
    expBRT = np.exp((bAdj - r) * T)
    
    if optType == 'call':
        price = S * expBRT * _normCdf(d1) - K * expRT * _normCdf(d2)
        delta = expBRT * _normCdf(d1)
    else:
        price = K * expRT * _normCdf(-d2) - S * expBRT * _normCdf(-d1)
        delta = -expBRT * _normCdf(-d1)
    
    h = 0.01 * S
    
    d1Up = (np.log((S + h) / K) + (bAdj + 0.5 * sigmaAdj**2) * T) / (sigmaAdj * sqrtT)
    d2Up = d1Up - sigmaAdj * sqrtT
    d1Down = (np.log((S - h) / K) + (bAdj + 0.5 * sigmaAdj**2) * T) / (sigmaAdj * sqrtT)
    d2Down = d1Down - sigmaAdj * sqrtT
    
    if optType == 'call':
        priceUp = (S + h) * expBRT * _normCdf(d1Up) - K * expRT * _normCdf(d2Up)
        priceDown = (S - h) * expBRT * _normCdf(d1Down) - K * expRT * _normCdf(d2Down)
    else:
        priceUp = K * expRT * _normCdf(-d2Up) - (S + h) * expBRT * _normCdf(-d1Up)
        priceDown = K * expRT * _normCdf(-d2Down) - (S - h) * expBRT * _normCdf(-d1Down)
    
    gamma = (priceUp - 2 * price + priceDown) / (h**2)
    
    hVol = 0.005
    sigmaAdjUp = (sigma + hVol) * np.sqrt((nSteps + 1) * (2 * nSteps + 1) / (6 * nSteps**2))
    bAdjUp = 0.5 * (b - 0.5 * (sigma + hVol)**2) + 0.5 * sigmaAdjUp**2
    d1VolUp = (np.log(S / K) + (bAdjUp + 0.5 * sigmaAdjUp**2) * T) / (sigmaAdjUp * sqrtT)
    d2VolUp = d1VolUp - sigmaAdjUp * sqrtT
    
    if optType == 'call':
        priceVolUp = S * np.exp((bAdjUp - r) * T) * _normCdf(d1VolUp) - K * expRT * _normCdf(d2VolUp)
    else:
        priceVolUp = K * expRT * _normCdf(-d2VolUp) - S * np.exp((bAdjUp - r) * T) * _normCdf(-d1VolUp)
    
    vega = (priceVolUp - price) / hVol
    
    hR = 0.0025
    expRTUp = np.exp(-(r + hR) * T)
    expBRTUp = np.exp((bAdj - (r + hR)) * T)
    
    if optType == 'call':
        priceRUp = S * expBRTUp * _normCdf(d1) - K * expRTUp * _normCdf(d2)
    else:
        priceRUp = K * expRTUp * _normCdf(-d2) - S * expBRTUp * _normCdf(-d1)
    
    rho = (priceRUp - price) / hR
    
    theta = 0.0
    if T > dt:
        TTheta = T - dt
        sqrtTTheta = np.sqrt(TTheta)
        d1Theta = (np.log(S / K) + (bAdj + 0.5 * sigmaAdj**2) * TTheta) / (sigmaAdj * sqrtTTheta)
        d2Theta = d1Theta - sigmaAdj * sqrtTTheta
        expRTTheta = np.exp(-r * TTheta)
        expBRTTheta = np.exp((bAdj - r) * TTheta)
        
        if optType == 'call':
            priceTheta = S * expBRTTheta * _normCdf(d1Theta) - K * expRTTheta * _normCdf(d2Theta)
        else:
            priceTheta = K * expRTTheta * _normCdf(-d2Theta) - S * expBRTTheta * _normCdf(-d1Theta)
        
        theta = (priceTheta - price) / dt
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta
    }

def binary(S, K, T, r, sigma, b=None, optType='call'):
    """
    Binary (cash-or-nothing) option pricing.
    
    Parameters:
        S: current asset price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        b: cost of carry (default r)
        optType: 'call' or 'put'
    
    Returns:
        dict with price, delta, gamma, vega, rho, theta
    """
    if b is None:
        b = r
    
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    
    expRT = np.exp(-r * T)
    
    if optType == 'call':
        price = expRT * _normCdf(d2)
        delta = expRT * _normPdf(d2) / (S * sigma * sqrtT)
        rho = -T * price
    else:
        price = expRT * _normCdf(-d2)
        delta = -expRT * _normPdf(d2) / (S * sigma * sqrtT)
        rho = -T * price
    
    gamma = -expRT * _normPdf(d2) * d1 / (S**2 * sigma**2 * T)
    vega = -expRT * _normPdf(d2) * d1 / sigma
    theta = -expRT * _normPdf(d2) * (r + ((-d1 * sigma) / (2 * T * sqrtT)))
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta
    }

def monteCarlo(S, K, T, r, sigma, b=None, nSims=10000, nSteps=100, optType='call', american=False, seed=None):
    """
    Monte Carlo option pricing for European and American options.
    
    Parameters:
        S: current asset price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        b: cost of carry (default r)
        nSims: number of simulations
        nSteps: number of time steps
        optType: 'call' or 'put'
        american: American style if True
        seed: random seed for reproducibility
    
    Returns:
        dict with price, stderr (standard error)
    """
    if b is None:
        b = r
    
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / nSteps
    drift = (b - 0.5 * sigma**2) * dt
    shock = sigma * np.sqrt(dt)
    
    paths = np.zeros((nSims, nSteps + 1))
    paths[:, 0] = S
    
    for t in range(1, nSteps + 1):
        z = np.random.standard_normal(nSims)
        paths[:, t] = paths[:, t - 1] * np.exp(drift + shock * z)
    
    if american:
        values = np.zeros((nSims, nSteps + 1))
        if optType == 'call':
            values[:, nSteps] = np.maximum(paths[:, nSteps] - K, 0)
        else:
            values[:, nSteps] = np.maximum(K - paths[:, nSteps], 0)
        
        for t in range(nSteps - 1, -1, -1):
            discValues = values[:, t + 1] * np.exp(-r * dt)
            
            if optType == 'call':
                intrinsic = np.maximum(paths[:, t] - K, 0)
            else:
                intrinsic = np.maximum(K - paths[:, t], 0)
            
            inMoney = intrinsic > 0
            
            if np.sum(inMoney) > 0:
                X = paths[inMoney, t].reshape(-1, 1)
                X = np.column_stack([np.ones(len(X)), X, X**2])
                y = discValues[inMoney]
                
                beta = np.linalg.lstsq(X, y, rcond=None)[0]
                continuation = X @ beta
                
                exercise = intrinsic[inMoney] > continuation
                values[inMoney, t] = np.where(exercise, intrinsic[inMoney], discValues[inMoney])
                values[~inMoney, t] = discValues[~inMoney]
            else:
                values[:, t] = discValues
        
        price = np.mean(values[:, 0])
    else:
        if optType == 'call':
            payoffs = np.maximum(paths[:, nSteps] - K, 0)
        else:
            payoffs = np.maximum(K - paths[:, nSteps], 0)
        
        price = np.exp(-r * T) * np.mean(payoffs)
    
    stderr = np.std(payoffs if not american else values[:, 0]) / np.sqrt(nSims)
    
    return {'price': price, 'stderr': stderr}

def impliedVol(price, S, K, T, r, optType='call', b=None, tol=1e-6, maxIter=100):
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
        price: observed option price
        S: current asset price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        optType: 'call' or 'put'
        b: cost of carry (default r)
        tol: tolerance for convergence
        maxIter: maximum iterations
    
    Returns:
        float: implied volatility
    """
    if b is None:
        b = r
    
    sigma = 0.3
    
    for i in range(maxIter):
        result = blackScholes(S, K, T, r, sigma, b, optType)
        diff = result['price'] - price
        
        if abs(diff) < tol:
            return sigma
        
        vega = result['vega']
        if vega < 1e-10:
            return np.nan
        
        sigma = sigma - diff / vega
        
        if sigma <= 0:
            sigma = 0.01
    
    return np.nan

def generateRange(modelFunc, paramRanges, fixedParams, optType='call'):
    """
    Generate option prices across parameter ranges.
    
    Parameters:
        modelFunc: pricing function (e.g., blackScholes)
        paramRanges: dict of {'param': {'start': val, 'end': val, 'step': val}}
        fixedParams: dict of fixed parameter values
        optType: 'call' or 'put'
    
    Returns:
        list of dicts with parameters and results
    """
    from itertools import product
    
    if not paramRanges:
        raise ValueError("At least one parameter range required")
    
    paramValues = {}
    for param, rangeDict in paramRanges.items():
        start = rangeDict['start']
        end = rangeDict['end']
        step = rangeDict['step']
        paramValues[param] = np.arange(start, end + step / 2, step)
    
    paramNames = list(paramValues.keys())
    combinations = list(product(*[paramValues[p] for p in paramNames]))
    
    results = []
    
    for combo in combinations:
        params = fixedParams.copy()
        for i, paramName in enumerate(paramNames):
            params[paramName] = combo[i]
        
        params['optType'] = optType
        
        try:
            optResult = modelFunc(**params)
            
            resultRow = {param: params[param] for param in paramNames}
            resultRow.update(optResult)
            
            results.append(resultRow)
        except Exception:
            pass
    
    return results