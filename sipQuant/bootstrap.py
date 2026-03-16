"""
sipQuant.bootstrap — Curve and surface construction.
Pure NumPy. No pandas, scipy, or external dependencies.

Functions
---------
forwardCurve         : Forward curve from spot, rates, storage costs, convenience yields.
discountCurve        : Discount curve from zero rates.
volSurface           : Volatility surface container.
interpVol            : Bilinear interpolation on a vol surface.
convenienceYieldCurve: Bootstrap convenience yields from futures prices.
bootstrapZeroCurve   : Bootstrap zero rates from par coupon bonds.
spreadCurve          : Add a basis spread to a rate curve.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Forward Curve
# ---------------------------------------------------------------------------

def forwardCurve(spotPrice, tenors, rates, storageCosts=None, convenienceYields=None):
    """Compute a cost-of-carry forward curve.

    F(t) = S * exp((r + c - y) * t)

    Parameters
    ----------
    spotPrice        : float, current spot price.
    tenors           : (m,) array of tenors in years.
    rates            : (m,) or scalar risk-free rates.
    storageCosts     : (m,) or scalar storage cost rates. Default 0.
    convenienceYields: (m,) or scalar convenience yield rates. Default 0.

    Returns
    -------
    dict: tenors (m,), forwards (m,), baseSpot (float).
    """
    tenors = np.asarray(tenors, dtype=float)
    rates = np.broadcast_to(np.asarray(rates, dtype=float), tenors.shape).copy()
    m = len(tenors)

    c = np.zeros(m)
    y = np.zeros(m)
    if storageCosts is not None:
        c = np.broadcast_to(np.asarray(storageCosts, dtype=float), tenors.shape).copy()
    if convenienceYields is not None:
        y = np.broadcast_to(np.asarray(convenienceYields, dtype=float), tenors.shape).copy()

    forwards = float(spotPrice) * np.exp((rates + c - y) * tenors)

    return {
        'tenors': tenors,
        'forwards': forwards,
        'baseSpot': float(spotPrice),
    }


# ---------------------------------------------------------------------------
# Discount Curve
# ---------------------------------------------------------------------------

def discountCurve(tenors, rates, method='linear'):
    """Discount curve from zero rates.

    Parameters
    ----------
    tenors : (m,) maturities in years.
    rates  : (m,) zero rates (continuously compounded).
    method : 'linear' | 'log_linear'. Interpolation method for intermediate
             tenors (stored for reference; the returned factors use the
             provided rates directly).

    Returns
    -------
    dict: tenors (m,), discountFactors (m,), zeroRates (m,), method (str).
    """
    tenors = np.asarray(tenors, dtype=float)
    rates = np.asarray(rates, dtype=float)

    if method == 'log_linear':
        # log_linear: linearly interpolate log discount factors.
        # For the given grid points, discount factors are exp(-r*t) in both cases.
        discountFactors = np.exp(-rates * tenors)
    else:
        # linear: linear interpolation of zero rates.
        discountFactors = np.exp(-rates * tenors)

    # Clip to (0, 1] for positive tenors; tenor=0 -> DF=1.
    discountFactors = np.clip(discountFactors, 1e-16, 1.0)

    return {
        'tenors': tenors,
        'discountFactors': discountFactors,
        'zeroRates': rates,
        'method': method,
    }


def _interpDiscountCurve(tenors, discountFactors, t, method='linear'):
    """Interpolate discount factor at maturity t."""
    tenors = np.asarray(tenors, dtype=float)
    discountFactors = np.asarray(discountFactors, dtype=float)

    if t <= tenors[0]:
        return float(discountFactors[0])
    if t >= tenors[-1]:
        return float(discountFactors[-1])

    idx = np.searchsorted(tenors, t) - 1
    t0, t1 = tenors[idx], tenors[idx + 1]
    df0, df1 = discountFactors[idx], discountFactors[idx + 1]
    frac = (t - t0) / (t1 - t0)

    if method == 'log_linear':
        return float(np.exp(np.log(df0) * (1.0 - frac) + np.log(df1) * frac))
    else:
        return float(df0 * (1.0 - frac) + df1 * frac)


# ---------------------------------------------------------------------------
# Vol Surface
# ---------------------------------------------------------------------------

def volSurface(strikes, tenors, vols):
    """Volatility surface container.

    Parameters
    ----------
    strikes : (nK,) strike array.
    tenors  : (nT,) tenor array.
    vols    : (nT x nK) implied vol matrix.

    Returns
    -------
    dict: strikes (nK,), tenors (nT,), vols (nT x nK), atmVols (nT,).
    """
    strikes = np.asarray(strikes, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    vols = np.asarray(vols, dtype=float)

    if vols.shape != (len(tenors), len(strikes)):
        raise ValueError(
            f"vols shape {vols.shape} does not match (nTenors={len(tenors)}, nStrikes={len(strikes)})."
        )

    # ATM vol: interpolate at the middle strike for each tenor.
    midStrikeIdx = len(strikes) // 2
    atmVols = vols[:, midStrikeIdx].copy()

    return {
        'strikes': strikes,
        'tenors': tenors,
        'vols': vols,
        'atmVols': atmVols,
    }


# ---------------------------------------------------------------------------
# Interpolate Vol
# ---------------------------------------------------------------------------

def interpVol(surface, strike, tenor):
    """Bilinear interpolation on a vol surface.

    Parameters
    ----------
    surface : dict as returned by volSurface.
    strike  : float.
    tenor   : float.

    Returns
    -------
    float: interpolated implied volatility.
    """
    strikes = surface['strikes']
    tenors = surface['tenors']
    vols = surface['vols']

    # Clamp to grid bounds.
    strike = float(np.clip(strike, strikes[0], strikes[-1]))
    tenor = float(np.clip(tenor, tenors[0], tenors[-1]))

    # Find bounding tenor indices.
    tIdx = np.searchsorted(tenors, tenor)
    tIdx = int(np.clip(tIdx, 1, len(tenors) - 1))
    t0, t1 = tenors[tIdx - 1], tenors[tIdx]
    wT1 = (tenor - t0) / (t1 - t0) if t1 > t0 else 0.0
    wT0 = 1.0 - wT1

    # Find bounding strike indices.
    kIdx = np.searchsorted(strikes, strike)
    kIdx = int(np.clip(kIdx, 1, len(strikes) - 1))
    k0, k1 = strikes[kIdx - 1], strikes[kIdx]
    wK1 = (strike - k0) / (k1 - k0) if k1 > k0 else 0.0
    wK0 = 1.0 - wK1

    # Bilinear interpolation.
    v00 = vols[tIdx - 1, kIdx - 1]
    v01 = vols[tIdx - 1, kIdx]
    v10 = vols[tIdx, kIdx - 1]
    v11 = vols[tIdx, kIdx]

    result = (wT0 * (wK0 * v00 + wK1 * v01)
              + wT1 * (wK0 * v10 + wK1 * v11))

    return float(result)


# ---------------------------------------------------------------------------
# Convenience Yield Curve
# ---------------------------------------------------------------------------

def convenienceYieldCurve(spotPrice, futuresPrices, tenors, rates):
    """Bootstrap convenience yields from observable futures prices.

    y(t) = r(t) - log(F(t) / S) / t

    Parameters
    ----------
    spotPrice    : float.
    futuresPrices: (m,) futures prices.
    tenors       : (m,) maturities in years (must be > 0).
    rates        : (m,) or scalar risk-free rates.

    Returns
    -------
    dict: tenors (m,), convenienceYields (m,), impliedForwards (m,).
    """
    spotPrice = float(spotPrice)
    futuresPrices = np.asarray(futuresPrices, dtype=float)
    tenors = np.asarray(tenors, dtype=float)
    rates = np.broadcast_to(np.asarray(rates, dtype=float), tenors.shape).copy()

    convYields = rates - np.log(futuresPrices / spotPrice + 1e-16) / (tenors + 1e-16)
    impliedForwards = spotPrice * np.exp(rates * tenors)

    return {
        'tenors': tenors,
        'convenienceYields': convYields,
        'impliedForwards': impliedForwards,
    }


# ---------------------------------------------------------------------------
# Bootstrap Zero Curve
# ---------------------------------------------------------------------------

def bootstrapZeroCurve(maturities, couponRates, faceValue=100.0):
    """Bootstrap zero rates from par coupon bonds (annual coupon assumed).

    For each maturity T_i:
      Price_i = faceValue (par bond, so coupon = coupon_rate * face)
      P_i = [faceValue - sum_{j<i} C * df_j] / (faceValue + C)

    Parameters
    ----------
    maturities  : (m,) sorted maturities in integer years (e.g. [1,2,3,...]).
    couponRates : (m,) annual coupon rates (e.g. [0.03, 0.035, ...]).
    faceValue   : float, default 100.

    Returns
    -------
    dict: maturities (m,), zeroRates (m,), discountFactors (m,).
    """
    maturities = np.asarray(maturities, dtype=float)
    couponRates = np.asarray(couponRates, dtype=float)
    m = len(maturities)
    dfs = np.zeros(m)
    zeroRates = np.zeros(m)
    fv = float(faceValue)

    for i in range(m):
        T = maturities[i]
        C = couponRates[i] * fv
        # Sum of discounted coupons for earlier maturities.
        pvCoupons = 0.0
        for j in range(i):
            pvCoupons += C * dfs[j]
        # Par price = faceValue.
        dfI = (fv - pvCoupons) / (fv + C + 1e-16)
        dfI = max(dfI, 1e-10)
        dfs[i] = dfI
        zeroRates[i] = -np.log(dfI) / T if T > 0 else 0.0

    return {
        'maturities': maturities,
        'zeroRates': zeroRates,
        'discountFactors': dfs,
    }


# ---------------------------------------------------------------------------
# Spread Curve
# ---------------------------------------------------------------------------

def spreadCurve(baseRates, spreadBps, tenors):
    """Add a basis spread to a base rate curve.

    Parameters
    ----------
    baseRates : (m,) or scalar base zero rates.
    spreadBps : (m,) or scalar spread in basis points.
    tenors    : (m,) tenor array.

    Returns
    -------
    dict: tenors (m,), adjustedRates (m,), baseRates (m,), spreadBps (m,).
    """
    tenors = np.asarray(tenors, dtype=float)
    m = len(tenors)
    baseRates = np.broadcast_to(np.asarray(baseRates, dtype=float), (m,)).copy()
    spreadBps = np.broadcast_to(np.asarray(spreadBps, dtype=float), (m,)).copy()

    adjustedRates = baseRates + spreadBps / 10000.0

    return {
        'tenors': tenors,
        'adjustedRates': adjustedRates,
        'baseRates': baseRates,
        'spreadBps': spreadBps,
    }
