import numpy as np

# ---------------------------------------------------------------------------
# sipQuant — liquidity.py
# Thin-market liquidity risk: LVAR, market impact, optimal execution,
# thin-market scoring, concentration risk, and liquidation cost estimation.
# Pure NumPy. No external dependencies.
# ---------------------------------------------------------------------------


def liquidityAdjustedVar(returns, volumes, alpha=0.05, spreadCost=None):
    """Liquidity-adjusted Value-at-Risk (LVAR) following Bangia et al.

    Parameters
    ----------
    returns : array-like of float
        Historical return series (proportional, not percentage).
    volumes : array-like of float
        Contemporaneous traded volumes corresponding to each return.
    alpha : float, optional
        Confidence level tail (e.g. 0.05 = 95% VaR). Default 0.05.
    spreadCost : float or None, optional
        If provided, the bid-ask spread cost as a fraction of position value.
        If None, estimated from volumes as:
        spreadCost = (1 / sqrt(vol / mean(vol))) * baseSpread,
        where baseSpread = 0.001.

    Returns
    -------
    dict
        Keys:

        - 'lvar' : float — liquidity-adjusted VaR (VaR + liquidity cost).
        - 'var' : float — standard historical VaR at confidence level.
        - 'liquidityCost' : float — 0.5 * spreadCost (Bangia et al. formula).
        - 'spreadCost' : float — spread cost used.

    Notes
    -----
    Historical VaR is taken as the alpha-quantile of the (negative) return
    distribution, i.e. the loss at the alpha percentile.
    LVAR = VaR + 0.5 * spreadCost. The 0.5 factor reflects that on average
    half the spread is paid on entry or exit.
    """
    returns = np.asarray(returns, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    var = float(-np.percentile(returns, alpha * 100))

    if spreadCost is None:
        meanVol = float(np.mean(volumes))
        if meanVol > 0 and len(volumes) > 0:
            latestVol = float(volumes[-1]) if len(volumes) > 0 else meanVol
            baseSpread = 0.001
            volRatio = latestVol / meanVol
            spreadCost = float((1.0 / np.sqrt(volRatio)) * baseSpread)
        else:
            spreadCost = 0.001

    spreadCost = float(spreadCost)
    liquidityCost = 0.5 * spreadCost
    lvar = var + liquidityCost

    return {
        'lvar': lvar,
        'var': var,
        'liquidityCost': liquidityCost,
        'spreadCost': spreadCost,
    }


def marketImpact(tradeSize, adv, model='almgren-chriss', eta=0.1, gamma=0.1):
    """Estimate market impact of a trade using standard impact models.

    Parameters
    ----------
    tradeSize : float
        Size of the trade (shares, contracts, tonnes, etc.).
    adv : float
        Average daily volume in the same units as tradeSize.
    model : str, optional
        Impact model to use. 'almgren-chriss' (default) or 'linear'.
    eta : float, optional
        Temporary impact coefficient. Default 0.1.
    gamma : float, optional
        Permanent impact coefficient. Default 0.1.

    Returns
    -------
    dict
        Keys:

        - 'permanentImpact' : float — lasting price impact from the trade.
        - 'temporaryImpact' : float — transient execution impact.
        - 'totalImpact' : float — sum of permanent and temporary impacts.
        - 'impactBps' : float — total impact in basis points
          (totalImpact * 10000).

    Notes
    -----
    Linear model: permanentImpact = gamma * (tradeSize/adv),
    temporaryImpact = eta * (tradeSize/adv).
    Almgren-Chriss: permanentImpact = gamma * (tradeSize/adv),
    temporaryImpact = eta * sqrt(tradeSize/adv).
    """
    tradeSize = float(tradeSize)
    adv = float(adv)
    eta = float(eta)
    gamma = float(gamma)

    if adv <= 0:
        raise ValueError(f"adv must be positive: got {adv}.")

    participation = tradeSize / adv

    if model == 'linear':
        permanentImpact = gamma * participation
        temporaryImpact = eta * participation
    elif model == 'almgren-chriss':
        permanentImpact = gamma * participation
        temporaryImpact = eta * float(np.sqrt(participation))
    else:
        raise ValueError(
            f"model must be 'almgren-chriss' or 'linear': got '{model}'."
        )

    totalImpact = permanentImpact + temporaryImpact

    return {
        'permanentImpact': permanentImpact,
        'temporaryImpact': temporaryImpact,
        'totalImpact': totalImpact,
        'impactBps': totalImpact * 10000.0,
    }


def optimalExecution(totalShares, T, adv, sigma, eta=0.1, gamma=0.1,
                     riskAversion=1e-6):
    """Compute the Almgren-Chriss optimal liquidation schedule.

    Parameters
    ----------
    totalShares : float
        Total position size to be unwound.
    T : int
        Total number of trading periods in the execution horizon.
    adv : float
        Average daily volume.
    sigma : float
        Daily price volatility (as a fraction, e.g. 0.02 for 2%).
    eta : float, optional
        Temporary impact coefficient. Default 0.1.
    gamma : float, optional
        Permanent impact coefficient. Default 0.1.
    riskAversion : float, optional
        Risk-aversion parameter lambda (trade-off between cost and risk).
        Default 1e-6.

    Returns
    -------
    dict
        Keys:

        - 'schedule' : np.ndarray — shares to trade in each period (length T).
        - 'trajectory' : np.ndarray — remaining shares at start of each period
          (length T+1, starts at totalShares, ends near zero).
        - 'expectedCost' : float — expected total execution cost.
        - 'expectedVariance' : float — variance of execution cost.
        - 'kappa' : float — decay rate of the optimal trajectory.

    Notes
    -----
    Analytical Almgren-Chriss solution:
    kappa = sqrt(riskAversion * sigma^2 / eta).
    Trajectory: x(t) = x0 * sinh(kappa*(T-t)) / sinh(kappa*T).
    Expected cost and variance use the standard AC closed-form expressions.
    """
    totalShares = float(totalShares)
    T = int(T)
    adv = float(adv)
    sigma = float(sigma)
    eta = float(eta)
    gamma = float(gamma)
    riskAversion = float(riskAversion)

    if T <= 0:
        raise ValueError(f"T must be a positive integer: got {T}.")
    if eta <= 0:
        raise ValueError(f"eta must be positive: got {eta}.")

    kappa = float(np.sqrt(riskAversion * sigma ** 2 / eta))

    # Trajectory: x(t) for t = 0, 1, ..., T
    t_vals = np.arange(T + 1, dtype=float)
    sinhKappaT = np.sinh(kappa * T)
    if sinhKappaT < 1e-12:
        # kappa near zero: uniform liquidation fallback
        trajectory = totalShares * (1.0 - t_vals / T)
    else:
        trajectory = totalShares * np.sinh(kappa * (T - t_vals)) / sinhKappaT

    # Schedule: shares traded in each period
    schedule = trajectory[:-1] - trajectory[1:]

    # Expected cost (AC approximation)
    expectedCost = float(
        0.5 * gamma * totalShares ** 2 / adv
        + eta * np.sum(schedule ** 2) / adv
    )

    # Expected variance
    expectedVariance = float(
        sigma ** 2 * np.sum(trajectory[:-1] ** 2)
    )

    return {
        'schedule': schedule,
        'trajectory': trajectory,
        'expectedCost': expectedCost,
        'expectedVariance': expectedVariance,
        'kappa': kappa,
    }


def thinMarketScore(tradeRecords, window=30):
    """Compute a thin-market liquidity score for a commodity market.

    Parameters
    ----------
    tradeRecords : list of dict
        List of TradeRecord dicts (from schema.py). Uses 'price' and 'volume'
        fields. All records are treated as falling within the window.
    window : int, optional
        Lookback window in days. Default 30.

    Returns
    -------
    dict
        Keys:

        - 'score' : float — liquidity score in [0, 1]. Higher = more liquid.
        - 'nTrades' : int — number of trades in the window.
        - 'avgVolume' : float — mean traded volume.
        - 'priceCV' : float — coefficient of variation of prices (std/mean).
        - 'window' : int — the window parameter used.

    Notes
    -----
    Score = (nTrades / window) * (1 - |priceCV|), clipped to [0, 1].
    priceCV = std(prices) / mean(prices). If mean(prices) is zero, priceCV=0.
    An empty trade record list returns score = 0.
    """
    window = int(window)

    if len(tradeRecords) == 0:
        return {
            'score': 0.0,
            'nTrades': 0,
            'avgVolume': 0.0,
            'priceCV': 0.0,
            'window': window,
        }

    prices = np.array([float(t['price']) for t in tradeRecords])
    volumes = np.array([float(t['volume']) for t in tradeRecords])
    nTrades = len(tradeRecords)
    avgVolume = float(np.mean(volumes))
    meanPrice = float(np.mean(prices))

    priceCV = float(np.std(prices) / meanPrice) if meanPrice > 0 else 0.0
    tradeFreq = nTrades / window
    score = float(np.clip(tradeFreq * (1.0 - abs(priceCV)), 0.0, 1.0))

    return {
        'score': score,
        'nTrades': nTrades,
        'avgVolume': avgVolume,
        'priceCV': priceCV,
        'window': window,
    }


def concentrationRisk(positions, volumes):
    """Herfindahl-Hirschman Index for position concentration.

    Parameters
    ----------
    positions : array-like of float
        Position sizes (can be signed; absolute values are used for HHI).
    volumes : array-like of float
        Market average daily volumes corresponding to each position.

    Returns
    -------
    dict
        Keys:

        - 'hhi' : float — Herfindahl-Hirschman Index in (0, 1].
          HHI = sum((|pos_i| / total_pos)^2).
        - 'participationRates' : np.ndarray — |pos_i| / vol_i for each
          position. inf where vol_i = 0.
        - 'concentrationScore' : float — normalised HHI; 1.0 = maximum
          concentration (single position dominates), 1/n = perfectly dispersed.

    Notes
    -----
    The HHI uses absolute position sizes to ensure the result is invariant
    to long/short direction. A value near 1/n (where n = number of positions)
    indicates a perfectly dispersed book.
    """
    positions = np.asarray(positions, dtype=float)
    volumes = np.asarray(volumes, dtype=float)

    absPosns = np.abs(positions)
    totalPos = float(np.sum(absPosns))

    if totalPos == 0.0:
        n = len(positions)
        return {
            'hhi': 0.0,
            'participationRates': np.zeros(n),
            'concentrationScore': 0.0,
        }

    shares = absPosns / totalPos
    hhi = float(np.sum(shares ** 2))

    with np.errstate(divide='ignore', invalid='ignore'):
        participationRates = np.where(volumes > 0, absPosns / volumes, np.inf)

    concentrationScore = hhi  # already in (1/n, 1]

    return {
        'hhi': hhi,
        'participationRates': participationRates,
        'concentrationScore': concentrationScore,
    }


def optimalLiquidation(position, adv, sigma, timeHorizon, costPerUnit=0.001):
    """Estimate TWAP/VWAP-style liquidation costs for a position.

    Parameters
    ----------
    position : float
        Total position size to liquidate.
    adv : float
        Average daily volume.
    sigma : float
        Daily price volatility (as a fraction, e.g. 0.02 for 2%).
    timeHorizon : int
        Number of trading periods over which to liquidate.
    costPerUnit : float, optional
        Fixed cost per unit traded (commissions, fees). Default 0.001.

    Returns
    -------
    dict
        Keys:

        - 'twapCost' : float — estimated total cost under TWAP execution
          (uniform schedule), including linear market impact and fixed costs.
        - 'vwapCost' : float — estimated total cost under VWAP execution
          (proportional to uniform ADV, identical to TWAP here as a lower
          bound; VWAP is modelled as TWAP * 0.9 for thin markets).
        - 'liquidationSchedule' : np.ndarray — uniform shares per period.
        - 'estimatedSlippage' : float — total linear market impact cost.
        - 'marketImpactCost' : float — per-period impact summed over horizon.

    Notes
    -----
    Linear market impact per period: eta * (sharesPerPeriod / adv) * sharesPerPeriod.
    Total slippage = sum of per-period impacts.
    VWAP cost is approximated as 90% of TWAP cost, reflecting the benefit of
    concentrating execution in high-volume windows.
    """
    position = float(position)
    adv = float(adv)
    sigma = float(sigma)
    timeHorizon = int(timeHorizon)
    costPerUnit = float(costPerUnit)

    if timeHorizon <= 0:
        raise ValueError(f"timeHorizon must be positive: got {timeHorizon}.")
    if adv <= 0:
        raise ValueError(f"adv must be positive: got {adv}.")

    sharesPerPeriod = abs(position) / timeHorizon
    liquidationSchedule = np.full(timeHorizon, sharesPerPeriod)

    # Linear impact per period: impact = eta * (shares/adv) * shares
    eta = 0.1
    perPeriodImpact = eta * (sharesPerPeriod / adv) * sharesPerPeriod
    marketImpactCost = perPeriodImpact * timeHorizon

    fixedCosts = abs(position) * costPerUnit
    twapCost = marketImpactCost + fixedCosts

    # VWAP approximated as 90% of TWAP cost
    vwapCost = twapCost * 0.9

    estimatedSlippage = marketImpactCost

    return {
        'twapCost': twapCost,
        'vwapCost': vwapCost,
        'liquidationSchedule': liquidationSchedule,
        'estimatedSlippage': estimatedSlippage,
        'marketImpactCost': marketImpactCost,
    }
