"""
sipQuant.commodity — Physical commodity pricing utilities.
SIP Global (Systematic Index Partners)

Pure NumPy. No external dependencies.

Functions
---------
seasonality          : STL-like decomposition: trend, seasonal, residual.
convenienceYield     : Implied convenience yield from spot and futures prices.
basis                : Cash-to-reference price differential.
gradeAdjustment      : Quality-adjusted price given grade factors.
transportDifferential: Delivered price with logistics cost breakdown.
localForwardCurve    : Local forward curve from spot, carry, and convenience yield.
rollingRollCost      : Roll cost computed from a forward curve.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Seasonality / STL-like decomposition
# ---------------------------------------------------------------------------

def seasonality(dates, values, period=52, method='stl'):
    """STL-like decomposition: trend (centred MA), seasonal (group means), residual.

    Parameters
    ----------
    dates  : array-like, length n — time index (used for ordering only).
    values : array-like, length n — observed values.
    period : int — seasonal period (default 52 for weekly data).
    method : str — 'stl' or 'additive' (both use the same additive algorithm).

    Returns
    -------
    dict with keys: trend, seasonal, residual, values, period, method.
    """
    values = np.asarray(values, dtype=float)
    n = len(values)

    # --- Trend: centred moving average of length = period ---
    half = period // 2
    trend = np.full(n, np.nan)
    for i in range(half, n - half):
        trend[i] = np.mean(values[i - half: i + half + 1])

    # Fill edges with nearest valid trend value
    if half > 0:
        trend[:half] = trend[half]
        trend[n - half:] = trend[n - half - 1]

    # --- Seasonal: group means of (values - trend) by position within period ---
    detrended = values - trend
    seasonal = np.zeros(n)
    for s in range(period):
        idx = np.arange(s, n, period)
        grpMean = np.nanmean(detrended[idx])
        seasonal[idx] = grpMean

    # Centre seasonal component so it sums to zero over one period
    seasonalAdj = seasonal - np.mean(seasonal[:period])
    seasonal = seasonalAdj

    residual = values - trend - seasonal

    return {
        'trend': trend,
        'seasonal': seasonal,
        'residual': residual,
        'values': values,
        'period': period,
        'method': method,
    }


# ---------------------------------------------------------------------------
# Convenience yield
# ---------------------------------------------------------------------------

def convenienceYield(spotPrice, futuresPrice, tenor, r, storageCost=0.0):
    """Implied convenience yield from cost-of-carry relationship.

    Formula: y = r + storageCost - log(F / S) / tenor

    Parameters
    ----------
    spotPrice    : float — current spot price S.
    futuresPrice : float — observed futures price F.
    tenor        : float — time to delivery in years.
    r            : float — risk-free rate (continuous).
    storageCost  : float — annualised storage cost rate.

    Returns
    -------
    dict with keys: convenienceYield, carryAdjustedForward, netCarry.
    """
    spotPrice = float(spotPrice)
    futuresPrice = float(futuresPrice)
    tenor = float(tenor)
    r = float(r)
    storageCost = float(storageCost)

    cy = r + storageCost - np.log(futuresPrice / spotPrice) / tenor
    carryAdjustedForward = spotPrice * np.exp((r + storageCost - cy) * tenor)
    netCarry = r + storageCost - cy

    return {
        'convenienceYield': float(cy),
        'carryAdjustedForward': float(carryAdjustedForward),
        'netCarry': float(netCarry),
    }


# ---------------------------------------------------------------------------
# Basis
# ---------------------------------------------------------------------------

def basis(cashPrice, referencePrice, market=None, grade=None):
    """Basis = cash price minus reference (benchmark) price.

    Parameters
    ----------
    cashPrice      : float — observed local cash price.
    referencePrice : float — benchmark / futures reference price.
    market         : str, optional — market label.
    grade          : str, optional — commodity grade label.

    Returns
    -------
    dict with keys: basis, basisBps, cashPrice, referencePrice, market, grade.
    """
    cashPrice = float(cashPrice)
    referencePrice = float(referencePrice)

    basisVal = cashPrice - referencePrice
    basisBps = (basisVal / referencePrice) * 10000.0 if referencePrice != 0.0 else np.nan

    return {
        'basis': float(basisVal),
        'basisBps': float(basisBps),
        'cashPrice': cashPrice,
        'referencePrice': referencePrice,
        'market': market,
        'grade': grade,
    }


# ---------------------------------------------------------------------------
# Grade adjustment
# ---------------------------------------------------------------------------

def gradeAdjustment(basePrice, gradeFactors):
    """Quality-adjusted price = basePrice + sum(gradeFactors).

    Parameters
    ----------
    basePrice    : float — unadjusted reference price.
    gradeFactors : dict or list/array of float adjustments.

    Returns
    -------
    dict with keys: adjustedPrice, totalAdjustment, gradeFactors.
    """
    basePrice = float(basePrice)

    if isinstance(gradeFactors, dict):
        factorValues = np.array(list(gradeFactors.values()), dtype=float)
    else:
        factorValues = np.asarray(gradeFactors, dtype=float)

    totalAdjustment = float(np.sum(factorValues))
    adjustedPrice = basePrice + totalAdjustment

    return {
        'adjustedPrice': adjustedPrice,
        'totalAdjustment': totalAdjustment,
        'gradeFactors': gradeFactors,
    }


# ---------------------------------------------------------------------------
# Transport differential
# ---------------------------------------------------------------------------

def transportDifferential(originPrice, freightCost, handlingCost=0.0, insuranceCost=0.0):
    """Delivered price from origin price plus logistics costs.

    deliveredPrice = originPrice + freightCost + handlingCost + insuranceCost

    Parameters
    ----------
    originPrice   : float — price at point of origin.
    freightCost   : float — freight / shipping cost.
    handlingCost  : float — terminal / handling charges.
    insuranceCost : float — cargo insurance cost.

    Returns
    -------
    dict with keys: deliveredPrice, originPrice, totalLogisticsCost, breakdown.
    """
    originPrice = float(originPrice)
    freightCost = float(freightCost)
    handlingCost = float(handlingCost)
    insuranceCost = float(insuranceCost)

    totalLogisticsCost = freightCost + handlingCost + insuranceCost
    deliveredPrice = originPrice + totalLogisticsCost

    breakdown = {
        'freight': freightCost,
        'handling': handlingCost,
        'insurance': insuranceCost,
    }

    return {
        'deliveredPrice': deliveredPrice,
        'originPrice': originPrice,
        'totalLogisticsCost': totalLogisticsCost,
        'breakdown': breakdown,
    }


# ---------------------------------------------------------------------------
# Local forward curve
# ---------------------------------------------------------------------------

def localForwardCurve(spotPrice, tenor, r, convYield, storageCost=0.0, basisAdjustment=0.0):
    """Local forward curve: F(t) = (S + basis) * exp((r + storage - convYield) * t).

    Parameters
    ----------
    spotPrice       : float — current spot price.
    tenor           : array-like — array of forward tenors in years.
    r               : float — risk-free rate.
    convYield       : float — convenience yield.
    storageCost     : float — annualised storage cost rate.
    basisAdjustment : float — local basis adjustment to spot.

    Returns
    -------
    dict with keys: tenors, forwards, netCarry, impliedConvenienceYield.
    """
    spotPrice = float(spotPrice)
    tenor = np.asarray(tenor, dtype=float)
    r = float(r)
    convYield = float(convYield)
    storageCost = float(storageCost)
    basisAdjustment = float(basisAdjustment)

    adjustedSpot = spotPrice + basisAdjustment
    netCarry = r + storageCost - convYield
    forwards = adjustedSpot * np.exp(netCarry * tenor)

    return {
        'tenors': tenor,
        'forwards': forwards,
        'netCarry': netCarry,
        'impliedConvenienceYield': convYield,
    }


# ---------------------------------------------------------------------------
# Rolling roll cost
# ---------------------------------------------------------------------------

def rollingRollCost(forwardCurve, rollDates):
    """Roll cost from a forward curve at specified roll indices.

    rollCosts[i] = forwards[rollDates[i] + 1] - forwards[rollDates[i]]

    Parameters
    ----------
    forwardCurve : dict — must contain 'forwards' (or 'prices') array and 'tenors'.
    rollDates    : array-like of int — indices into the forwards array at which rolls occur.

    Returns
    -------
    dict with keys: rollCosts, totalRollCost, annualizedRollCost.
    """
    rollDates = np.asarray(rollDates, dtype=int)

    if 'forwards' in forwardCurve:
        fwd = np.asarray(forwardCurve['forwards'], dtype=float)
    elif 'prices' in forwardCurve:
        fwd = np.asarray(forwardCurve['prices'], dtype=float)
    else:
        raise KeyError("forwardCurve must contain 'forwards' or 'prices' key.")

    tenors = np.asarray(forwardCurve.get('tenors', np.arange(len(fwd))), dtype=float)

    rollCosts = np.array(
        [fwd[rd + 1] - fwd[rd] for rd in rollDates if rd + 1 < len(fwd)],
        dtype=float,
    )

    totalRollCost = float(np.sum(rollCosts))

    # Annualise: divide by total tenor span if available
    tenorSpan = float(tenors[-1] - tenors[0]) if len(tenors) > 1 else 1.0
    annualizedRollCost = totalRollCost / tenorSpan if tenorSpan > 0 else totalRollCost

    return {
        'rollCosts': rollCosts,
        'totalRollCost': totalRollCost,
        'annualizedRollCost': annualizedRollCost,
    }
