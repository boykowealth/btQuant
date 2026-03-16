"""
sipQuant.otc — OTC instrument pricing for physical commodity markets.
SIP Global (Systematic Index Partners)

Pure NumPy. No external dependencies.

Functions
---------
commoditySwap  : Fixed-float commodity swap NPV and Greeks.
asianSwap      : Asian (average price) swap NPV.
collar         : Long cap + short floor using Black-Scholes analytics.
physicalForward: PV of a physically-settled forward contract.
swaption       : Black's formula swaption on forward swap rate.
asianOption    : Monte Carlo Asian option with antithetic variates.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normCdf(x):
    """Cumulative distribution function of the standard normal.
    Abramowitz & Stegun polynomial approximation, max error 7.5e-8.
    """
    x = np.asarray(x, dtype=float)
    xAbs = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * xAbs)
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    p = 1.0 - _normPdf(xAbs) * poly
    return np.where(x >= 0, p, 1.0 - p)


def _normPdf(x):
    """Probability density function of the standard normal."""
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Commodity swap
# ---------------------------------------------------------------------------

def commoditySwap(fixedPrice, indexCurve, notional, schedule, r):
    """Fixed-float commodity swap NPV.

    NPV = sum_i [ (indexCurve[i] - fixedPrice) * notional * DF(t_i) ]
    where DF(t_i) = exp(-r * t_i).

    Parameters
    ----------
    fixedPrice : float — fixed price leg.
    indexCurve : array-like, length n — floating index prices at each payment date.
    notional   : float — contract notional.
    schedule   : array-like, length n — payment tenors in years.
    r          : float — flat discount rate (continuous).

    Returns
    -------
    dict with keys: npv, fixedLegPV, floatLegPV, greeks: {delta, dv01}.
    """
    indexCurve = np.asarray(indexCurve, dtype=float)
    schedule = np.asarray(schedule, dtype=float)
    fixedPrice = float(fixedPrice)
    notional = float(notional)
    r = float(r)

    df = np.exp(-r * schedule)
    floatLegPV = float(np.sum(indexCurve * notional * df))
    fixedLegPV = float(np.sum(fixedPrice * notional * df))
    npv = floatLegPV - fixedLegPV

    delta = float(np.sum(notional * df))
    dv01 = float(np.sum(-schedule * notional * (indexCurve - fixedPrice) * df) * 0.0001)

    return {
        'npv': npv,
        'fixedLegPV': fixedLegPV,
        'floatLegPV': floatLegPV,
        'greeks': {
            'delta': delta,
            'dv01': dv01,
        },
    }


# ---------------------------------------------------------------------------
# Asian swap
# ---------------------------------------------------------------------------

def asianSwap(fixedPrice, indexPrices, notional, r, T):
    """Asian (arithmetic average price) swap NPV.

    NPV = (mean(indexPrices) - fixedPrice) * notional * exp(-r * T)

    Parameters
    ----------
    fixedPrice   : float — fixed price leg.
    indexPrices  : array-like — observed or projected index prices over the averaging period.
    notional     : float — contract notional.
    r            : float — discount rate.
    T            : float — maturity in years.

    Returns
    -------
    dict with keys: npv, averageIndex, fixedPrice, dv01.
    """
    indexPrices = np.asarray(indexPrices, dtype=float)
    fixedPrice = float(fixedPrice)
    notional = float(notional)
    r = float(r)
    T = float(T)

    averageIndex = float(np.mean(indexPrices))
    df = np.exp(-r * T)
    npv = (averageIndex - fixedPrice) * notional * df
    dv01 = -T * npv * 0.0001

    return {
        'npv': float(npv),
        'averageIndex': averageIndex,
        'fixedPrice': fixedPrice,
        'dv01': float(dv01),
    }


# ---------------------------------------------------------------------------
# Collar
# ---------------------------------------------------------------------------

def collar(S, capStrike, floorStrike, T, r, sigma, notional=1.0, q=0.0):
    """Long cap (call) + short floor (put) using Black-Scholes analytics.

    Net collar price = capPrice - floorPrice.

    Parameters
    ----------
    S          : float — spot price.
    capStrike  : float — cap (call) strike.
    floorStrike: float — floor (put) strike.
    T          : float — time to expiry in years.
    r          : float — risk-free rate.
    sigma      : float — volatility.
    notional   : float — contract notional.
    q          : float — continuous dividend / convenience yield.

    Returns
    -------
    dict with keys: price, capPrice, floorPrice, greeks: {delta, gamma, vega, theta, rho}.
    """
    S = float(S)
    r = float(r)
    sigma = float(sigma)
    T = float(T)
    q = float(q)
    notional = float(notional)

    sqrtT = np.sqrt(T)

    def _bsPrice(K, optType):
        K = float(K)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        if optType == 'call':
            price = (S * np.exp(-q * T) * _normCdf(d1)
                     - K * np.exp(-r * T) * _normCdf(d2))
        else:
            price = (K * np.exp(-r * T) * _normCdf(-d2)
                     - S * np.exp(-q * T) * _normCdf(-d1))
        return price, d1, d2

    capPrice, d1Cap, d2Cap = _bsPrice(capStrike, 'call')
    floorPrice, d1Floor, d2Floor = _bsPrice(floorStrike, 'put')

    netPrice = (capPrice - floorPrice) * notional

    # Greeks (net collar = long call + short put)
    deltaCall = float(np.exp(-q * T) * _normCdf(d1Cap))
    deltaPut = float(-np.exp(-q * T) * _normCdf(-d1Floor))
    delta = (deltaCall - (-deltaPut)) * notional  # long call - short put

    gammaCall = float(np.exp(-q * T) * _normPdf(d1Cap) / (S * sigma * sqrtT))
    gammaPut = float(np.exp(-q * T) * _normPdf(d1Floor) / (S * sigma * sqrtT))
    gamma = (gammaCall - gammaPut) * notional

    vegaCall = float(S * np.exp(-q * T) * _normPdf(d1Cap) * sqrtT)
    vegaPut = float(S * np.exp(-q * T) * _normPdf(d1Floor) * sqrtT)
    vega = (vegaCall - vegaPut) * notional

    thetaCall = float(
        (-S * np.exp(-q * T) * _normPdf(d1Cap) * sigma / (2 * sqrtT)
         - r * float(capStrike) * np.exp(-r * T) * _normCdf(d2Cap)
         + q * S * np.exp(-q * T) * _normCdf(d1Cap)) / 365.0
    )
    thetaPut = float(
        (-S * np.exp(-q * T) * _normPdf(d1Floor) * sigma / (2 * sqrtT)
         + r * float(floorStrike) * np.exp(-r * T) * _normCdf(-d2Floor)
         - q * S * np.exp(-q * T) * _normCdf(-d1Floor)) / 365.0
    )
    theta = (thetaCall - thetaPut) * notional

    rhoCall = float(float(capStrike) * T * np.exp(-r * T) * _normCdf(d2Cap) / 100.0)
    rhoPut = float(-float(floorStrike) * T * np.exp(-r * T) * _normCdf(-d2Floor) / 100.0)
    rho = (rhoCall - rhoPut) * notional

    return {
        'price': float(netPrice),
        'capPrice': float(capPrice * notional),
        'floorPrice': float(floorPrice * notional),
        'greeks': {
            'delta': float(delta),
            'gamma': float(gamma),
            'vega': float(vega),
            'theta': float(theta),
            'rho': float(rho),
        },
    }


# ---------------------------------------------------------------------------
# Physical forward
# ---------------------------------------------------------------------------

def physicalForward(F, deliveryTenor, r, storageCost=0.0, qualityPremium=0.0, notional=1.0):
    """Present value of a physically-settled forward contract.

    PV = (F + qualityPremium) * exp(-(r + storageCost) * T) * notional

    Parameters
    ----------
    F              : float — forward / futures price.
    deliveryTenor  : float — time to delivery in years.
    r              : float — risk-free rate.
    storageCost    : float — annualised storage cost rate.
    qualityPremium : float — quality / grade premium on top of F.
    notional       : float — contract notional.

    Returns
    -------
    dict with keys: pv, forwardPrice, adjustedForward, greeks: {delta, dv01}.
    """
    F = float(F)
    T = float(deliveryTenor)
    r = float(r)
    storageCost = float(storageCost)
    qualityPremium = float(qualityPremium)
    notional = float(notional)

    adjustedForward = F + qualityPremium
    df = np.exp(-(r + storageCost) * T)
    pv = adjustedForward * df * notional

    delta = float(df * notional)
    dv01 = float(-T * pv * 0.0001)

    return {
        'pv': float(pv),
        'forwardPrice': F,
        'adjustedForward': adjustedForward,
        'greeks': {
            'delta': delta,
            'dv01': dv01,
        },
    }


# ---------------------------------------------------------------------------
# Swaption (Black's formula)
# ---------------------------------------------------------------------------

def swaption(fixedPrice, indexCurve, notional, schedule, r, sigma, T, optType='call'):
    """Swaption priced with Black's formula on the forward swap rate.

    forwardSwapRate = sum(indexCurve[i] * DF(t_i)) / annuity
    annuity = sum(DF(t_i))

    Parameters
    ----------
    fixedPrice : float — fixed rate / price of the underlying swap.
    indexCurve : array-like — forward index prices.
    notional   : float — notional.
    schedule   : array-like — payment tenors in years.
    r          : float — discount rate.
    sigma      : float — Black volatility.
    T          : float — swaption expiry in years (T <= schedule[0]).
    optType    : str — 'call' (payer) or 'put' (receiver).

    Returns
    -------
    dict with keys: price, forwardSwapRate, annuity, greeks: {delta, vega, theta}.
    """
    indexCurve = np.asarray(indexCurve, dtype=float)
    schedule = np.asarray(schedule, dtype=float)
    fixedPrice = float(fixedPrice)
    notional = float(notional)
    r = float(r)
    sigma = float(sigma)
    T = float(T)

    df = np.exp(-r * schedule)
    annuity = float(np.sum(df))
    forwardSwapRate = float(np.sum(indexCurve * df) / annuity)

    F = forwardSwapRate
    K = fixedPrice
    sqrtT = np.sqrt(T)

    if T <= 0 or sigma <= 0:
        price = max(F - K, 0.0) * annuity * notional if optType == 'call' else max(K - F, 0.0) * annuity * notional
        return {
            'price': float(price),
            'forwardSwapRate': F,
            'annuity': annuity,
            'greeks': {'delta': 0.0, 'vega': 0.0, 'theta': 0.0},
        }

    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    if optType == 'call':
        price = annuity * notional * (F * _normCdf(d1) - K * _normCdf(d2))
        delta = annuity * notional * _normCdf(d1)
    else:
        price = annuity * notional * (K * _normCdf(-d2) - F * _normCdf(-d1))
        delta = -annuity * notional * _normCdf(-d1)

    vega = float(annuity * notional * F * _normPdf(d1) * sqrtT)
    theta = float(-annuity * notional * F * _normPdf(d1) * sigma / (2.0 * sqrtT) / 365.0)

    return {
        'price': float(price),
        'forwardSwapRate': F,
        'annuity': annuity,
        'greeks': {
            'delta': float(delta),
            'vega': vega,
            'theta': theta,
        },
    }


# ---------------------------------------------------------------------------
# Asian option (Monte Carlo)
# ---------------------------------------------------------------------------

def asianOption(S, K, T, r, sigma, nSims=10000, nSteps=50, optType='call', q=0.0, seed=None):
    """Arithmetic-average Asian option priced by Monte Carlo with antithetic variates.

    Parameters
    ----------
    S       : float — spot price.
    K       : float — strike price.
    T       : float — maturity in years.
    r       : float — risk-free rate.
    sigma   : float — volatility.
    nSims   : int   — number of simulation paths (antithetic: nSims/2 base + nSims/2 mirror).
    nSteps  : int   — number of averaging time steps.
    optType : str   — 'call' or 'put'.
    q       : float — continuous dividend / convenience yield.
    seed    : int or None — RNG seed for reproducibility.

    Returns
    -------
    dict with keys: price, stderr, greeks: {delta, vega}.
    """
    S = float(S)
    K = float(K)
    T = float(T)
    r = float(r)
    sigma = float(sigma)
    q = float(q)

    rng = np.random.default_rng(seed)
    dt = T / nSteps
    halfSims = nSims // 2

    # Standard normal draws: (halfSims, nSteps)
    Z = rng.standard_normal((halfSims, nSteps))
    Zanti = -Z

    def _simulatePaths(zMatrix):
        logS = np.log(S) + np.cumsum(
            (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * zMatrix,
            axis=1,
        )
        return np.exp(logS)  # (halfSims, nSteps)

    paths = _simulatePaths(Z)
    pathsAnti = _simulatePaths(Zanti)

    avgPrice = np.mean(paths, axis=1)
    avgPriceAnti = np.mean(pathsAnti, axis=1)

    if optType == 'call':
        payoff = np.maximum(avgPrice - K, 0.0)
        payoffAnti = np.maximum(avgPriceAnti - K, 0.0)
    else:
        payoff = np.maximum(K - avgPrice, 0.0)
        payoffAnti = np.maximum(K - avgPriceAnti, 0.0)

    combinedPayoff = 0.5 * (payoff + payoffAnti)
    df = np.exp(-r * T)
    discounted = df * combinedPayoff
    price = float(np.mean(discounted))
    stderr = float(np.std(discounted) / np.sqrt(halfSims))

    # Delta via finite difference
    bumpS = S * 1.01
    pathsBump = _simulatePaths(Z) * (bumpS / S)  # scale existing paths
    avgBump = np.mean(pathsBump, axis=1)
    if optType == 'call':
        payoffBump = np.maximum(avgBump - K, 0.0)
    else:
        payoffBump = np.maximum(K - avgBump, 0.0)
    priceBump = float(np.mean(df * payoffBump))
    delta = (priceBump - price) / (S * 0.01)

    # Vega via finite difference
    bumpSigma = sigma + 0.01
    logSVega = np.log(S) + np.cumsum(
        (r - q - 0.5 * bumpSigma ** 2) * dt + bumpSigma * np.sqrt(dt) * Z,
        axis=1,
    )
    pathsVega = np.exp(logSVega)
    avgVega = np.mean(pathsVega, axis=1)
    if optType == 'call':
        payoffVega = np.maximum(avgVega - K, 0.0)
    else:
        payoffVega = np.maximum(K - avgVega, 0.0)
    priceVega = float(np.mean(df * payoffVega))
    vega = (priceVega - price) / 0.01

    return {
        'price': price,
        'stderr': stderr,
        'greeks': {
            'delta': float(delta),
            'vega': float(vega),
        },
    }
