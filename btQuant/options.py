import numpy as np

def _normCdf(x):
    """
    Approximate cumulative distribution function of standard normal distribution.
    
    Parameters:
        x: input value
    
    Returns:
        float: CDF value at x
    """
    return 0.5 * (1.0 + np.tanh(x / np.sqrt(2.0) * 0.7978845608))

def _normPdf(x):
    """
    Probability density function of standard normal distribution.
    
    Parameters:
        x: input value
    
    Returns:
        float: PDF value at x
    """
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def blackScholes(S, K, T, r, sigma, q=0.0, optType='call'):
    """
    Black-Scholes option pricing model with Greeks.
    
    Parameters:
        S: current stock price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        q: dividend yield
        optType: 'call' or 'put'
    
    Returns:
        dict: price, delta, gamma, vega, rho, theta
    """
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    
    expQT = np.exp(-q * T)
    expRT = np.exp(-r * T)
    
    if optType == 'call':
        price = S * expQT * _normCdf(d1) - K * expRT * _normCdf(d2)
        delta = expQT * _normCdf(d1)
        rho = K * T * expRT * _normCdf(d2)
        theta = (-S * expQT * _normPdf(d1) * sigma / (2 * sqrtT) + 
                q * S * expQT * _normCdf(d1) - r * K * expRT * _normCdf(d2))
    else:
        price = K * expRT * _normCdf(-d2) - S * expQT * _normCdf(-d1)
        delta = -expQT * _normCdf(-d1)
        rho = -K * T * expRT * _normCdf(-d2)
        theta = (-S * expQT * _normPdf(d1) * sigma / (2 * sqrtT) - 
                q * S * expQT * _normCdf(-d1) + r * K * expRT * _normCdf(-d2))
    
    gamma = expQT * _normPdf(d1) / (S * sigma * sqrtT)
    vega = S * expQT * _normPdf(d1) * sqrtT
    
    return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega, 'rho': rho, 'theta': theta}


def binary(S, K, T, r, sigma, q=0.0, optType='call'):
    """
    Binary (digital) option pricing with Greeks.
    
    Parameters:
        S: current stock price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        q: dividend yield
        optType: 'call' or 'put'
    
    Returns:
        dict: price, delta, gamma, vega, rho, theta
    """
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
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
    
    return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega, 'rho': rho, 'theta': theta}


def spread(S1, S2, K, T, r, sigma1, sigma2, rho, q1=0.0, q2=0.0, optType='call'):
    """
    Spread option pricing (option on the difference between two assets).
    
    Parameters:
        S1: current price of asset 1
        S2: current price of asset 2
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma1: volatility of asset 1
        sigma2: volatility of asset 2
        rho: correlation between assets
        q1: dividend yield of asset 1
        q2: dividend yield of asset 2
        optType: 'call' or 'put'
    
    Returns:
        dict: price, delta1, delta2, gamma1, gamma2, vega1, vega2
    """
    F1 = S1 * np.exp((r - q1) * T)
    F2 = S2 * np.exp((r - q2) * T)
    F2K = F2 + K
    
    if F2K <= 0:
        if optType == 'call':
            return {'price': max(F1 - F2 - K, 0) * np.exp(-r * T), 
                   'delta1': 0, 'delta2': 0, 'gamma1': 0, 'gamma2': 0, 'vega1': 0, 'vega2': 0}
        else:
            return {'price': 0, 'delta1': 0, 'delta2': 0, 'gamma1': 0, 'gamma2': 0, 'vega1': 0, 'vega2': 0}
    
    sigma = np.sqrt(sigma1**2 - 2 * rho * sigma1 * sigma2 * F2 / F2K + (sigma2 * F2 / F2K)**2)
    sqrtT = np.sqrt(T)
    d1 = (np.log(F1 / F2K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    expRT = np.exp(-r * T)
    
    if optType == 'call':
        price = expRT * (F1 * _normCdf(d1) - F2K * _normCdf(d2))
        delta1 = expRT * _normCdf(d1)
        delta2 = -expRT * _normCdf(d2)
    else:
        price = expRT * (F2K * _normCdf(-d2) - F1 * _normCdf(-d1))
        delta1 = -expRT * _normCdf(-d1)
        delta2 = expRT * _normCdf(-d2)
    
    gamma1 = expRT * _normPdf(d1) / (F1 * sigma * sqrtT)
    gamma2 = expRT * _normPdf(d1) / (F2K * sigma * sqrtT)
    vega1 = expRT * F1 * _normPdf(d1) * sqrtT
    vega2 = expRT * F2 * _normPdf(d1) * sqrtT
    
    return {'price': price, 'delta1': delta1, 'delta2': delta2, 
            'gamma1': gamma1, 'gamma2': gamma2, 'vega1': vega1, 'vega2': vega2}


def barrier(S, K, T, r, sigma, barrierLevel, q=0.0, optType='call', barrierType='down-and-out', rebate=0.0):
    """
    Barrier option pricing (option that activates or deactivates at a barrier level).
    
    Parameters:
        S: current stock price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        barrierLevel: barrier price level
        q: dividend yield
        optType: 'call' or 'put'
        barrierType: 'down-and-out', 'down-and-in', 'up-and-out', 'up-and-in'
        rebate: rebate payment if barrier is hit
    
    Returns:
        dict: price, delta, gamma, vega
    """
    mu = (r - q - 0.5 * sigma**2) / sigma**2
    sqrtT = np.sqrt(T)
    y = np.log(barrierLevel**2 / (S * K)) / (sigma * sqrtT)
    z = np.log(barrierLevel / S) / (sigma * sqrtT)
    expRT = np.exp(-r * T)
    expQT = np.exp(-q * T)
    vanilla = blackScholes(S, K, T, r, sigma, q, optType)
    
    if barrierType == 'down-and-out':
        if optType == 'call':
            if K >= barrierLevel:
                price = vanilla['price'] - S * expQT * (barrierLevel / S)**(2 * mu) * \
                       (_normCdf(y) - _normCdf(y - sigma * sqrtT)) + \
                       K * expRT * (barrierLevel / S)**(2 * mu - 2) * _normCdf(y - sigma * sqrtT)
            else:
                price = 0.0
        else:
            if K <= barrierLevel:
                price = 0.0
            else:
                price = vanilla['price'] + S * expQT * (barrierLevel / S)**(2 * mu) * \
                       _normCdf(-y + sigma * sqrtT) - K * expRT * (barrierLevel / S)**(2 * mu - 2) * _normCdf(-y)
    
    elif barrierType == 'down-and-in':
        if optType == 'call':
            if K >= barrierLevel:
                dio = vanilla['price'] - S * expQT * (barrierLevel / S)**(2 * mu) * \
                      (_normCdf(y) - _normCdf(y - sigma * sqrtT)) + \
                      K * expRT * (barrierLevel / S)**(2 * mu - 2) * _normCdf(y - sigma * sqrtT)
                price = vanilla['price'] - dio
            else:
                price = vanilla['price']
        else:
            if K <= barrierLevel:
                price = vanilla['price']
            else:
                dio = vanilla['price'] + S * expQT * (barrierLevel / S)**(2 * mu) * \
                      _normCdf(-y + sigma * sqrtT) - K * expRT * (barrierLevel / S)**(2 * mu - 2) * _normCdf(-y)
                price = vanilla['price'] - dio
    
    elif barrierType == 'up-and-out':
        if optType == 'call':
            if K >= barrierLevel:
                price = 0.0
            else:
                price = vanilla['price'] + S * expQT * (barrierLevel / S)**(2 * mu) * \
                       _normCdf(y - sigma * sqrtT) - K * expRT * (barrierLevel / S)**(2 * mu - 2) * _normCdf(y)
        else:
            if K <= barrierLevel:
                price = vanilla['price'] - S * expQT * (barrierLevel / S)**(2 * mu) * \
                       (_normCdf(-y) - _normCdf(-y + sigma * sqrtT)) + \
                       K * expRT * (barrierLevel / S)**(2 * mu - 2) * _normCdf(-y + sigma * sqrtT)
            else:
                price = 0.0
    
    else:
        if optType == 'call':
            if K >= barrierLevel:
                price = vanilla['price']
            else:
                uio = vanilla['price'] + S * expQT * (barrierLevel / S)**(2 * mu) * \
                      _normCdf(y - sigma * sqrtT) - K * expRT * (barrierLevel / S)**(2 * mu - 2) * _normCdf(y)
                price = vanilla['price'] - uio
        else:
            if K <= barrierLevel:
                uio = vanilla['price'] - S * expQT * (barrierLevel / S)**(2 * mu) * \
                      (_normCdf(-y) - _normCdf(-y + sigma * sqrtT)) + \
                      K * expRT * (barrierLevel / S)**(2 * mu - 2) * _normCdf(-y + sigma * sqrtT)
                price = vanilla['price'] - uio
            else:
                price = vanilla['price']
    
    if rebate > 0:
        rebate_value = rebate * expRT * _normCdf(z if 'down' in barrierType else -z)
        price += rebate_value
    
    return {'price': price, 'delta': 0.0, 'gamma': 0.0, 'vega': 0.0}


def asian(S, K, T, r, sigma, q=0.0, nSteps=100, optType='call', avgType='geometric'):
    """
    Asian option pricing (option on average price).
    
    Parameters:
        S: current stock price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        q: dividend yield
        nSteps: number of averaging steps
        optType: 'call' or 'put'
        avgType: 'geometric' or 'arithmetic'
    
    Returns:
        dict: price, delta, gamma, vega, rho, theta (or price, stderr for arithmetic)
    """
    if avgType == 'geometric':
        dt = T / nSteps
        sigmaAdj = sigma * np.sqrt((nSteps + 1) * (2 * nSteps + 1) / (6 * nSteps**2))
        muAdj = 0.5 * (r - q - 0.5 * sigma**2)
        rAdj = r - q - muAdj + 0.5 * sigmaAdj**2
        qAdj = r - rAdj
        
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (rAdj - qAdj + 0.5 * sigmaAdj**2) * T) / (sigmaAdj * sqrtT)
        d2 = d1 - sigmaAdj * sqrtT
        expRT = np.exp(-r * T)
        expQAdjT = np.exp(-qAdj * T)
        
        if optType == 'call':
            price = S * expQAdjT * _normCdf(d1) - K * expRT * _normCdf(d2)
            delta = expQAdjT * _normCdf(d1)
        else:
            price = K * expRT * _normCdf(-d2) - S * expQAdjT * _normCdf(-d1)
            delta = -expQAdjT * _normCdf(-d1)
        
        h = 0.01 * S
        d1Up = (np.log((S + h) / K) + (rAdj - qAdj + 0.5 * sigmaAdj**2) * T) / (sigmaAdj * sqrtT)
        d2Up = d1Up - sigmaAdj * sqrtT
        d1Down = (np.log((S - h) / K) + (rAdj - qAdj + 0.5 * sigmaAdj**2) * T) / (sigmaAdj * sqrtT)
        d2Down = d1Down - sigmaAdj * sqrtT
        
        if optType == 'call':
            priceUp = (S + h) * expQAdjT * _normCdf(d1Up) - K * expRT * _normCdf(d2Up)
            priceDown = (S - h) * expQAdjT * _normCdf(d1Down) - K * expRT * _normCdf(d2Down)
        else:
            priceUp = K * expRT * _normCdf(-d2Up) - (S + h) * expQAdjT * _normCdf(-d1Up)
            priceDown = K * expRT * _normCdf(-d2Down) - (S - h) * expQAdjT * _normCdf(-d1Down)
        
        gamma = (priceUp - 2 * price + priceDown) / (h**2)
        
        hVol = 0.005
        sigmaAdjUp = (sigma + hVol) * np.sqrt((nSteps + 1) * (2 * nSteps + 1) / (6 * nSteps**2))
        d1VolUp = (np.log(S / K) + (rAdj - qAdj + 0.5 * sigmaAdjUp**2) * T) / (sigmaAdjUp * sqrtT)
        d2VolUp = d1VolUp - sigmaAdjUp * sqrtT
        
        if optType == 'call':
            priceVolUp = S * np.exp(-qAdj * T) * _normCdf(d1VolUp) - K * expRT * _normCdf(d2VolUp)
        else:
            priceVolUp = K * expRT * _normCdf(-d2VolUp) - S * np.exp(-qAdj * T) * _normCdf(-d1VolUp)
        
        vega = (priceVolUp - price) / hVol
        
        hR = 0.0025
        expRTUp = np.exp(-(r + hR) * T)
        if optType == 'call':
            priceRUp = S * expQAdjT * _normCdf(d1) - K * expRTUp * _normCdf(d2)
        else:
            priceRUp = K * expRTUp * _normCdf(-d2) - S * expQAdjT * _normCdf(-d1)
        
        rho = (priceRUp - price) / hR
        
        theta = 0.0
        if T > dt:
            TTheta = T - dt
            sqrtTTheta = np.sqrt(TTheta)
            d1Theta = (np.log(S / K) + (rAdj - qAdj + 0.5 * sigmaAdj**2) * TTheta) / (sigmaAdj * sqrtTTheta)
            d2Theta = d1Theta - sigmaAdj * sqrtTTheta
            expRTTheta = np.exp(-r * TTheta)
            expQAdjTTheta = np.exp(-qAdj * TTheta)
            
            if optType == 'call':
                priceTheta = S * expQAdjTTheta * _normCdf(d1Theta) - K * expRTTheta * _normCdf(d2Theta)
            else:
                priceTheta = K * expRTTheta * _normCdf(-d2Theta) - S * expQAdjTTheta * _normCdf(-d1Theta)
            
            theta = (priceTheta - price) / dt
        
        return {'price': price, 'delta': delta, 'gamma': gamma, 'vega': vega, 'rho': rho, 'theta': theta}
    else:
        return {'price': np.nan, 'stderr': np.nan}


def binomial(S, K, T, r, sigma, q=0.0, N=100, optType='call', american=False):
    """
    Binomial tree option pricing model.
    
    Parameters:
        S: current stock price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        q: dividend yield
        N: number of time steps
        optType: 'call' or 'put'
        american: True for American options, False for European
    
    Returns:
        dict: price, delta, gamma, theta
    """
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
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
            values[j, i] = disc * (p * values[j, i + 1] + (1 - p) * values[j + 1, i + 1])
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
                    (values[1, 2] - values[2, 2]) / (prices[1, 2] - prices[2, 2])) / \
                    ((prices[0, 2] - prices[2, 2]) / 2)
        else:
            gamma = 0.0
    else:
        delta = 0.0
        gamma = 0.0
    
    theta = (values[1, 1] - price) / dt if N > 0 else 0.0
    
    return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta}


def trinomial(S, K, T, r, sigma, q=0.0, N=50, optType='call', american=False):
    """
    Trinomial tree option pricing model.
    
    Parameters:
        S: current stock price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        sigma: volatility
        q: dividend yield
        N: number of time steps
        optType: 'call' or 'put'
        american: True for American options, False for European
    
    Returns:
        dict: price, delta, gamma, theta
    """
    dt = T / N
    dx = sigma * np.sqrt(3 * dt)
    nu = r - q - 0.5 * sigma**2
    
    pu = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 + nu * dt / dx)
    pd = 0.5 * ((sigma**2 * dt + nu**2 * dt**2) / dx**2 - nu * dt / dx)
    pm = 1.0 - pu - pd
    disc = np.exp(-r * dt)
    
    M = 2 * N + 1
    assetPrices = S * np.exp(np.arange(-N, N + 1) * dx)
    
    if optType == 'call':
        values = np.maximum(assetPrices - K, 0)
    else:
        values = np.maximum(K - assetPrices, 0)
    
    for step in range(N):
        newValues = np.zeros(M)
        for j in range(1, M - 1):
            newValues[j] = disc * (pu * values[j + 1] + pm * values[j] + pd * values[j - 1])
            
            if american:
                currentPrice = assetPrices[j]
                if optType == 'call':
                    newValues[j] = max(newValues[j], currentPrice - K)
                else:
                    newValues[j] = max(newValues[j], K - currentPrice)
        
        values = newValues.copy()
    
    price = values[N]
    delta = (values[N + 1] - values[N - 1]) / (assetPrices[N + 1] - assetPrices[N - 1])
    gamma = ((values[N + 1] - values[N]) / (assetPrices[N + 1] - assetPrices[N]) - 
            (values[N] - values[N - 1]) / (assetPrices[N] - assetPrices[N - 1])) / \
            ((assetPrices[N + 1] - assetPrices[N - 1]) / 2)
    theta = 0.0
    
    return {'price': price, 'delta': delta, 'gamma': gamma, 'theta': theta}


def simulate(pricingModel, paths, r, T, **modelParams):
    """
    Monte Carlo simulation wrapper for option pricing.
    
    Parameters:
        pricingModel: pricing function to use
        paths: simulated price paths (nSims x nSteps)
        r: risk-free rate
        T: time to maturity (years)
        **modelParams: additional parameters for pricing model
    
    Returns:
        dict: price, stderr
    """
    nSims = paths.shape[0]
    payoffs = np.zeros(nSims)
    
    for i in range(nSims):
        result = pricingModel(S=paths[i, -1], T=T, r=r, **modelParams)
        payoffs[i] = result['price']
    
    price = np.exp(-r * T) * np.mean(payoffs)
    stderr = np.std(payoffs) / np.sqrt(nSims)
    
    return {'price': price, 'stderr': stderr}


def impliedVol(price, S, K, T, r, optType='call', q=0.0, tol=1e-6, maxIter=100):
    """
    Calculate implied volatility using Newton-Raphson method.
    
    Parameters:
        price: observed option price
        S: current stock price
        K: strike price
        T: time to maturity (years)
        r: risk-free rate
        optType: 'call' or 'put'
        q: dividend yield
        tol: convergence tolerance
        maxIter: maximum iterations
    
    Returns:
        float: implied volatility (or np.nan if not converged)
    """
    sigma = 0.3
    
    for i in range(maxIter):
        result = blackScholes(S, K, T, r, sigma, q, optType)
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


def buildForwardCurve(spotPrice, tenors, rates, storageCosts=None, convenienceYields=None):
    """
    Build forward curve for commodities or other assets.
    
    Parameters:
        spotPrice: current spot price
        tenors: time to maturity array (years)
        rates: risk-free rates array
        storageCosts: storage cost rates array
        convenienceYields: convenience yield rates array
    
    Returns:
        array: forward prices
    """
    if storageCosts is None:
        storageCosts = np.zeros_like(tenors)
    if convenienceYields is None:
        convenienceYields = np.zeros_like(tenors)
    
    forwards = spotPrice * np.exp((rates + storageCosts - convenienceYields) * tenors)
    return forwards


def bootstrapCurve(spotPrice, futuresPrices, tenors, assumedRate=0.05):
    """
    Bootstrap convenience yields from futures prices.
    
    Parameters:
        spotPrice: current spot price
        futuresPrices: observed futures prices array
        tenors: time to maturity array (years)
        assumedRate: assumed risk-free rate
    
    Returns:
        dict: convenience_yields, storage_costs
    """
    convenienceYields = assumedRate - np.log(futuresPrices / spotPrice) / tenors
    storageCosts = np.zeros_like(tenors)
    
    return {'convenience_yields': convenienceYields, 'storage_costs': storageCosts}