import numpy as np

# ---------------------------------------------------------------------------
# sipQuant — book.py
# OTC dealer book aggregation: net Greeks, hedge ratios, P&L attribution,
# scenario analysis, book summary, and margin estimation.
# Pure NumPy. No external dependencies.
# ---------------------------------------------------------------------------

_LONG_DIRECTIONS = frozenset({'buy', 'long', 'receive_fixed'})
_SHORT_DIRECTIONS = frozenset({'sell', 'short', 'pay_fixed'})


def _directionSign(direction):
    """Return +1 for long/buy/receive_fixed, -1 for short/sell/pay_fixed."""
    d = str(direction).lower()
    if d in _LONG_DIRECTIONS:
        return 1.0
    if d in _SHORT_DIRECTIONS:
        return -1.0
    raise ValueError(f"Unrecognised direction '{direction}'.")


def netGreeks(positions):
    """Aggregate Greeks across all OTC positions in the book.

    Parameters
    ----------
    positions : list of dict
        List of OTCPosition dicts (from schema.py). Each must have a 'greeks'
        dict with keys 'delta', 'gamma', 'vega', 'theta', 'rho', a 'notional'
        float, and a 'direction' string.

    Returns
    -------
    dict
        Keys: 'delta', 'gamma', 'vega', 'theta', 'rho' — net signed Greeks
        aggregated across all positions, scaled by notional and direction.

    Notes
    -----
    Direction sign convention: buy/long/receive_fixed = +1,
    sell/short/pay_fixed = -1.
    Each Greek contribution = greek * notional * direction_sign.
    """
    delta = 0.0
    gamma = 0.0
    vega = 0.0
    theta = 0.0
    rho = 0.0

    for pos in positions:
        sign = _directionSign(pos['direction'])
        notional = float(pos['notional'])
        g = pos['greeks']
        delta += g['delta'] * notional * sign
        gamma += g['gamma'] * notional * sign
        vega += g['vega'] * notional * sign
        theta += g['theta'] * notional * sign
        rho += g['rho'] * notional * sign

    return {
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'theta': theta,
        'rho': rho,
    }


def hedgeRatios(positions, hedgeInstrumentDelta):
    """Compute number of hedge instruments needed to delta-hedge the book.

    Parameters
    ----------
    positions : list of dict
        List of OTCPosition dicts (from schema.py).
    hedgeInstrumentDelta : float
        Delta of one unit of the hedge instrument (e.g. 1.0 for a futures
        contract, 0.5 for an at-the-money option).

    Returns
    -------
    dict
        Keys:

        - 'hedgeUnits' : float — number of hedge instruments to transact
          (negative = sell hedge, positive = buy hedge).
        - 'netDelta' : float — book net delta before hedging.
        - 'residualDelta' : float — delta remaining after rounding hedgeUnits
          to nearest integer.

    Notes
    -----
    hedgeUnits = -netDelta / hedgeInstrumentDelta.
    residualDelta is computed using the rounded integer hedge units.
    """
    ng = netGreeks(positions)
    netDelta = ng['delta']
    hedgeInstrumentDelta = float(hedgeInstrumentDelta)

    if hedgeInstrumentDelta == 0.0:
        raise ValueError("hedgeInstrumentDelta cannot be zero.")

    hedgeUnits = -netDelta / hedgeInstrumentDelta
    roundedUnits = float(np.round(hedgeUnits))
    residualDelta = netDelta + roundedUnits * hedgeInstrumentDelta

    return {
        'hedgeUnits': hedgeUnits,
        'netDelta': netDelta,
        'residualDelta': residualDelta,
    }


def pnlAttribution(positions, priceMoves, volMoves=None, timeDecay=None):
    """P&L attribution using a first-order Taylor expansion per Greek component.

    dPnL = delta*dS + 0.5*gamma*dS^2 + vega*dVol + theta*dT

    Parameters
    ----------
    positions : list of dict
        List of OTCPosition dicts (from schema.py). Each position must carry
        a 'marketId' key or will default to 'default'.
    priceMoves : dict
        Mapping of {marketId: dS} price changes. Positions without a matching
        marketId key fall back to key 'default' if present, else dS=0.
    volMoves : dict, optional
        Mapping of {marketId: dVol} implied-vol changes. Same fallback logic.
        If None, vega contribution is zero.
    timeDecay : float, optional
        Number of calendar days elapsed. If None, theta contribution is zero.

    Returns
    -------
    dict
        Keys:

        - 'totalPnL' : float
        - 'deltaComponent' : float
        - 'gammaComponent' : float
        - 'vegaComponent' : float
        - 'thetaComponent' : float
        - 'totalByPosition' : np.ndarray — per-position total P&L.
    """
    if volMoves is None:
        volMoves = {}
    if timeDecay is None:
        timeDecay = 0.0

    timeDecay = float(timeDecay)

    deltaComp = 0.0
    gammaComp = 0.0
    vegaComp = 0.0
    thetaComp = 0.0
    byPosition = np.zeros(len(positions))

    for i, pos in enumerate(positions):
        sign = _directionSign(pos['direction'])
        notional = float(pos['notional'])
        g = pos['greeks']
        marketId = pos.get('marketId', 'default')

        dS = priceMoves.get(marketId, priceMoves.get('default', 0.0))
        dVol = volMoves.get(marketId, volMoves.get('default', 0.0))

        d = sign * notional * g['delta'] * dS
        gm = sign * notional * 0.5 * g['gamma'] * dS ** 2
        v = sign * notional * g['vega'] * dVol
        th = sign * notional * g['theta'] * timeDecay

        deltaComp += d
        gammaComp += gm
        vegaComp += v
        thetaComp += th
        byPosition[i] = d + gm + v + th

    totalPnL = deltaComp + gammaComp + vegaComp + thetaComp

    return {
        'totalPnL': totalPnL,
        'deltaComponent': deltaComp,
        'gammaComponent': gammaComp,
        'vegaComponent': vegaComp,
        'thetaComponent': thetaComp,
        'totalByPosition': byPosition,
    }


def scenarioShock(positions, scenarios):
    """Apply scenario shocks to the book and estimate P&L impact.

    Parameters
    ----------
    positions : list of dict
        List of OTCPosition dicts (from schema.py).
    scenarios : list of dict
        Each scenario dict must contain:

        - 'name' : str — scenario label.
        - 'priceShock' : float — uniform price change dS applied to all markets.
        - 'volShock' : float — uniform implied-vol change dVol applied to all
          markets.

    Returns
    -------
    dict
        Keys:

        - 'scenarioResults' : list of dicts, one per scenario, each with keys
          'name', 'pnl', 'deltaContrib', 'vegaContrib'.

    Notes
    -----
    Gamma contribution is included in the per-scenario 'pnl' via the Taylor
    expansion but is not separately reported. Delta and vega contributions are
    isolated for transparency.
    """
    results = []

    for scenario in scenarios:
        name = scenario['name']
        dS = float(scenario.get('priceShock', 0.0))
        dVol = float(scenario.get('volShock', 0.0))

        deltaCont = 0.0
        vegaCont = 0.0
        totalPnL = 0.0

        for pos in positions:
            sign = _directionSign(pos['direction'])
            notional = float(pos['notional'])
            g = pos['greeks']

            d = sign * notional * g['delta'] * dS
            gm = sign * notional * 0.5 * g['gamma'] * dS ** 2
            v = sign * notional * g['vega'] * dVol

            deltaCont += d
            vegaCont += v
            totalPnL += d + gm + v

        results.append({
            'name': name,
            'pnl': totalPnL,
            'deltaContrib': deltaCont,
            'vegaContrib': vegaCont,
        })

    return {'scenarioResults': results}


def bookSummary(positions):
    """Summarize book composition by instrument type and direction.

    Parameters
    ----------
    positions : list of dict
        List of OTCPosition dicts (from schema.py).

    Returns
    -------
    dict
        Keys:

        - 'totalNotional' : float — sum of all position notionals.
        - 'nPositions' : int — number of positions in the book.
        - 'byInstrument' : dict — mapping of instrumentType to total notional.
        - 'netGreeks' : dict — aggregated Greeks (see netGreeks).
        - 'concentrationRisk' : float — maximum single-position fraction of
          total notional (0 if book is empty).
    """
    if len(positions) == 0:
        return {
            'totalNotional': 0.0,
            'nPositions': 0,
            'byInstrument': {},
            'netGreeks': {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0,
                          'theta': 0.0, 'rho': 0.0},
            'concentrationRisk': 0.0,
        }

    notionals = np.array([float(p['notional']) for p in positions])
    totalNotional = float(np.sum(notionals))

    byInstrument = {}
    for pos in positions:
        iType = str(pos.get('instrumentType', 'unknown'))
        byInstrument[iType] = byInstrument.get(iType, 0.0) + float(pos['notional'])

    concentrationRisk = float(np.max(notionals) / totalNotional) if totalNotional > 0 else 0.0

    return {
        'totalNotional': totalNotional,
        'nPositions': len(positions),
        'byInstrument': byInstrument,
        'netGreeks': netGreeks(positions),
        'concentrationRisk': concentrationRisk,
    }


def marginEstimate(positions, initialMarginRate=0.1, variationMarginBuffer=0.05):
    """Estimate initial and variation margin requirements for the book.

    Parameters
    ----------
    positions : list of dict
        List of OTCPosition dicts (from schema.py).
    initialMarginRate : float, optional
        Fraction of notional applied to initial margin per unit of delta.
        Default 0.10 (10%).
    variationMarginBuffer : float, optional
        Buffer multiplier applied to daily theta decay for variation margin.
        Default 0.05 (5%).

    Returns
    -------
    dict
        Keys:

        - 'initialMargin' : float — sum(notional * initialMarginRate * |delta|).
        - 'variationMargin' : float — sum(|theta/252| * notional) *
          variationMarginBuffer.
        - 'totalMargin' : float — initialMargin + variationMargin.

    Notes
    -----
    The daily theta decay uses a 252-trading-day convention.
    Greeks are taken directly from each position's 'greeks' dict without
    direction scaling — margin is always a positive quantity.
    """
    initialMargin = 0.0
    variationMargin = 0.0

    for pos in positions:
        notional = float(pos['notional'])
        g = pos['greeks']
        initialMargin += notional * initialMarginRate * abs(g['delta'])
        variationMargin += abs(g['theta'] / 252.0) * notional

    variationMargin *= variationMarginBuffer
    totalMargin = initialMargin + variationMargin

    return {
        'initialMargin': initialMargin,
        'variationMargin': variationMargin,
        'totalMargin': totalMargin,
    }
