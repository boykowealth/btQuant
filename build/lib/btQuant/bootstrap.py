import numpy as np

def curve(x, y, z=None, method='cubic', nPoints=100):
    """
    Bootstrap curves using interpolation.
    
    Parameters:
        x: maturities/tenors (1D array)
        y: rates/prices (1D array)
        z: optional cash flows/coupons (1D array)
        method: 'linear', 'cubic', 'pchip'
        nPoints: number of interpolation points
    
    Returns:
        dict with 'x' and 'y' arrays
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if method == 'linear':
        xNew = np.linspace(np.min(x), np.max(x), nPoints)
        yNew = np.interp(xNew, x, y)
    
    elif method == 'cubic':
        xNew = np.linspace(np.min(x), np.max(x), nPoints)
        yNew = _cubicSpline(x, y, xNew)
    
    elif method == 'pchip':
        xNew = np.linspace(np.min(x), np.max(x), nPoints)
        yNew = _pchip(x, y, xNew)
    
    else:
        raise ValueError("method must be 'linear', 'cubic', or 'pchip'")
    
    return {'x': xNew, 'y': yNew}

def _cubicSpline(x, y, xNew):
    """Cubic spline interpolation."""
    n = len(x)
    h = np.diff(x)
    
    alpha = np.zeros(n - 1)
    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (y[i + 1] - y[i]) - (3 / h[i - 1]) * (y[i] - y[i - 1])
    
    l = np.ones(n)
    mu = np.zeros(n)
    z = np.zeros(n)
    
    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]
    
    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)
    
    for j in range(n - 2, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])
    
    yNew = np.zeros(len(xNew))
    for i, xi in enumerate(xNew):
        j = np.searchsorted(x[1:], xi)
        j = min(j, n - 2)
        
        dx = xi - x[j]
        yNew[i] = y[j] + b[j] * dx + c[j] * dx**2 + d[j] * dx**3
    
    return yNew

def _pchip(x, y, xNew):
    """Piecewise Cubic Hermite Interpolating Polynomial."""
    n = len(x)
    h = np.diff(x)
    delta = np.diff(y) / h
    
    m = np.zeros(n)
    
    m[0] = delta[0]
    m[-1] = delta[-1]
    
    for i in range(1, n - 1):
        if delta[i - 1] * delta[i] > 0:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]
            m[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i])
        else:
            m[i] = 0
    
    yNew = np.zeros(len(xNew))
    for i, xi in enumerate(xNew):
        j = np.searchsorted(x[1:], xi)
        j = min(j, n - 2)
        
        t = (xi - x[j]) / h[j]
        yNew[i] = (y[j] * (1 + 2 * t) * (1 - t)**2 + 
                   y[j + 1] * (3 - 2 * t) * t**2 + 
                   m[j] * h[j] * t * (1 - t)**2 + 
                   m[j + 1] * h[j] * t**2 * (t - 1))
    
    return yNew

def surface(x, y, z, method='linear', gridSize=(100, 100)):
    """
    Bootstrap 2D surfaces (e.g., volatility surfaces).
    
    Parameters:
        x: first dimension (e.g., maturities)
        y: second dimension (e.g., strikes)
        z: values at (x, y) points
        method: 'linear', 'cubic'
        gridSize: (nx, ny) tuple
    
    Returns:
        dict with 'X', 'Y', 'Z' arrays
    """
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    
    xGrid = np.linspace(np.min(x), np.max(x), gridSize[0])
    yGrid = np.linspace(np.min(y), np.max(y), gridSize[1])
    X, Y = np.meshgrid(xGrid, yGrid)
    
    if method == 'linear':
        Z = _bilinearInterp(x, y, z, X.flatten(), Y.flatten())
        Z = Z.reshape(gridSize[1], gridSize[0])
    
    elif method == 'cubic':
        Z = _bicubicInterp(x, y, z, X.flatten(), Y.flatten())
        Z = Z.reshape(gridSize[1], gridSize[0])
    
    else:
        raise ValueError("method must be 'linear' or 'cubic'")
    
    return {'X': X, 'Y': Y, 'Z': Z}

def _bilinearInterp(x, y, z, xi, yi):
    """Bilinear interpolation for scattered data."""
    zi = np.zeros(len(xi))
    
    for i in range(len(xi)):
        dists = np.sqrt((x - xi[i])**2 + (y - yi[i])**2)
        
        if np.min(dists) < 1e-10:
            zi[i] = z[np.argmin(dists)]
        else:
            weights = 1.0 / (dists + 1e-10)
            weights /= np.sum(weights)
            zi[i] = np.sum(weights * z)
    
    return zi

def _bicubicInterp(x, y, z, xi, yi):
    """Bicubic interpolation approximation."""
    return _bilinearInterp(x, y, z, xi, yi)

def zeroRateCurve(maturities, prices, method='linear', nPoints=100):
    """
    Bootstrap zero rate curve from bond prices.
    
    Parameters:
        maturities: bond maturities (years)
        prices: bond prices
        method: interpolation method
        nPoints: number of points
    
    Returns:
        dict with 'maturities' and 'zeroRates'
    """
    maturities = np.asarray(maturities)
    prices = np.asarray(prices)
    
    zeroRates = -np.log(prices) / maturities
    
    return curve(maturities, zeroRates, method=method, nPoints=nPoints)

def forwardCurve(maturities, zeroRates, method='linear', nPoints=100):
    """
    Derive forward rate curve from zero rates.
    
    Parameters:
        maturities: maturities
        zeroRates: zero rates
        method: interpolation method
        nPoints: number of points
    
    Returns:
        dict with 'maturities' and 'forwardRates'
    """
    maturities = np.asarray(maturities)
    zeroRates = np.asarray(zeroRates)
    
    forwardRates = np.zeros(len(maturities) - 1)
    
    for i in range(len(maturities) - 1):
        t1 = maturities[i]
        t2 = maturities[i + 1]
        r1 = zeroRates[i]
        r2 = zeroRates[i + 1]
        
        forwardRates[i] = (r2 * t2 - r1 * t1) / (t2 - t1)
    
    forwardMaturities = (maturities[:-1] + maturities[1:]) / 2
    
    return curve(forwardMaturities, forwardRates, method=method, nPoints=nPoints)

def discountCurve(maturities, zeroRates, method='linear', nPoints=100):
    """
    Derive discount factor curve from zero rates.
    
    Parameters:
        maturities: maturities
        zeroRates: zero rates
        method: interpolation method
        nPoints: number of points
    
    Returns:
        dict with 'maturities' and 'discountFactors'
    """
    maturities = np.asarray(maturities)
    zeroRates = np.asarray(zeroRates)
    
    discountFactors = np.exp(-zeroRates * maturities)
    
    return curve(maturities, discountFactors, method=method, nPoints=nPoints)

def yieldCurve(maturities, prices, coupons=None, method='linear', nPoints=100):
    """
    Bootstrap yield curve from bond data.
    
    Parameters:
        maturities: bond maturities
        prices: bond prices
        coupons: coupon rates (optional)
        method: interpolation method
        nPoints: number of points
    
    Returns:
        dict with 'maturities' and 'yields'
    """
    maturities = np.asarray(maturities)
    prices = np.asarray(prices)
    
    if coupons is None:
        coupons = np.zeros(len(maturities))
    else:
        coupons = np.asarray(coupons)
    
    yields = np.zeros(len(maturities))
    
    for i in range(len(maturities)):
        ytm = coupons[i]
        
        for _ in range(50):
            pv = sum(coupons[i] / (1 + ytm)**t for t in range(1, int(maturities[i]) + 1))
            pv += 1.0 / (1 + ytm)**maturities[i]
            
            diff = pv - prices[i]
            
            if abs(diff) < 1e-6:
                break
            
            dur = sum(t * coupons[i] / (1 + ytm)**(t + 1) for t in range(1, int(maturities[i]) + 1))
            dur += maturities[i] / (1 + ytm)**(maturities[i] + 1)
            
            ytm -= diff / dur
        
        yields[i] = ytm
    
    return curve(maturities, yields, method=method, nPoints=nPoints)

def volSurface(maturities, strikes, impliedVols, method='linear', gridSize=(50, 50)):
    """
    Bootstrap implied volatility surface.
    
    Parameters:
        maturities: option maturities
        strikes: strike prices
        impliedVols: implied volatilities
        method: interpolation method
        gridSize: grid dimensions
    
    Returns:
        dict with 'maturities', 'strikes', 'vols'
    """
    return surface(maturities, strikes, impliedVols, method=method, gridSize=gridSize)

def creditCurve(maturities, cdsSpreads, recoveryRate=0.4, method='linear', nPoints=100):
    """
    Bootstrap credit default swap curve.
    
    Parameters:
        maturities: CDS maturities
        cdsSpreads: CDS spreads (in basis points)
        recoveryRate: recovery rate assumption
        method: interpolation method
        nPoints: number of points
    
    Returns:
        dict with 'maturities' and 'defaultProbabilities'
    """
    maturities = np.asarray(maturities)
    cdsSpreads = np.asarray(cdsSpreads) / 10000
    
    hazardRates = cdsSpreads / (1 - recoveryRate)
    
    defaultProbs = 1 - np.exp(-hazardRates * maturities)
    
    return curve(maturities, defaultProbs, method=method, nPoints=nPoints)

def fxForwardCurve(spot, domesticRates, foreignRates, maturities, method='linear', nPoints=100):
    """
    Bootstrap FX forward curve.
    
    Parameters:
        spot: spot FX rate
        domesticRates: domestic interest rates
        foreignRates: foreign interest rates
        maturities: forward maturities
        method: interpolation method
        nPoints: number of points
    
    Returns:
        dict with 'maturities' and 'forwardRates'
    """
    domesticRates = np.asarray(domesticRates)
    foreignRates = np.asarray(foreignRates)
    maturities = np.asarray(maturities)
    
    forwardRates = spot * np.exp((domesticRates - foreignRates) * maturities)
    
    return curve(maturities, forwardRates, method=method, nPoints=nPoints)

def inflationCurve(maturities, breakevens, realRates, method='linear', nPoints=100):
    """
    Bootstrap inflation curve from breakeven rates.
    
    Parameters:
        maturities: maturities
        breakevens: breakeven inflation rates
        realRates: real interest rates
        method: interpolation method
        nPoints: number of points
    
    Returns:
        dict with 'maturities' and 'inflationRates'
    """
    breakevens = np.asarray(breakevens)
    realRates = np.asarray(realRates)
    maturities = np.asarray(maturities)
    
    inflationRates = breakevens - realRates
    
    return curve(maturities, inflationRates, method=method, nPoints=nPoints)