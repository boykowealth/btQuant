import numpy as np

# ---------------------------------------------------------------------------
# sipQuant — index.py
# IOSCO-aligned commodity index infrastructure: calculation, audit trail,
# restatement, roll schedule, proxy regression, and backtesting.
# Pure NumPy. No external dependencies.
# ---------------------------------------------------------------------------

_RESTATEMENT_COUNTER = [0]  # module-level mutable counter for restatementId


def calculateIndex(tradeRecords, indexSpec, calculationDate):
    """Calculate index value for a given date.

    Parameters
    ----------
    tradeRecords : list of dict
        List of TradeRecord dicts (from schema.py). Each must have keys:
        'date', 'price', 'volume', 'grade' (or 'market' as constituent key).
    indexSpec : dict
        IndexSpec dict (from schema.py) with keys 'constituents',
        'weightsMethod', 'name', 'version', 'methodology' (optional).
    calculationDate : comparable
        Only trades on or before this date are included.

    Returns
    -------
    dict
        Keys:

        - 'indexValue' : float — weighted sum of constituent VWAPs.
        - 'constituentValues' : dict — VWAP per constituent.
        - 'constituentWeights' : dict — weight applied to each constituent.
        - 'nTrades' : int — total number of trades used.
        - 'calculationDate' : same type as input.
        - 'methodology' : str — from indexSpec or 'not_specified'.

    Notes
    -----
    Filtering uses the 'grade' field of each TradeRecord as the constituent
    identifier. If a constituent has no matching trades, its VWAP is 0.
    Steps:

    1. Filter trades to date <= calculationDate AND grade in constituents.
    2. Compute VWAP per constituent.
    3. Apply weighting ('equal' or 'volume').
    4. Index value = weighted sum of constituent VWAPs.
    """
    constituents = indexSpec['constituents']
    weightsMethod = indexSpec['weightsMethod']
    methodology = indexSpec.get('methodology', indexSpec.get('rollRule', 'not_specified'))

    # Step 1 — filter
    filtered = [
        t for t in tradeRecords
        if t['date'] <= calculationDate and t['grade'] in constituents
    ]

    # Step 2 — VWAP per constituent
    constituentPrices = {}
    constituentVolumes = {}
    for c in constituents:
        ctrades = [t for t in filtered if t['grade'] == c]
        if ctrades:
            prices = np.array([t['price'] for t in ctrades])
            volumes = np.array([t['volume'] for t in ctrades])
            vwap = float(np.sum(prices * volumes) / np.sum(volumes))
            constituentPrices[c] = vwap
            constituentVolumes[c] = float(np.sum(volumes))
        else:
            constituentPrices[c] = 0.0
            constituentVolumes[c] = 0.0

    # Step 3 — weights
    constituentWeights = {}
    if weightsMethod == 'equal':
        w = 1.0 / len(constituents) if constituents else 0.0
        for c in constituents:
            constituentWeights[c] = w
    elif weightsMethod in ('volume', 'liquidity'):
        totalVol = sum(constituentVolumes.values())
        for c in constituents:
            constituentWeights[c] = (
                constituentVolumes[c] / totalVol if totalVol > 0 else 0.0
            )
    else:
        # 'custom' — fall back to equal
        w = 1.0 / len(constituents) if constituents else 0.0
        for c in constituents:
            constituentWeights[c] = w

    # Step 4 — index value
    indexValue = float(sum(
        constituentWeights[c] * constituentPrices[c] for c in constituents
    ))

    return {
        'indexValue': indexValue,
        'constituentValues': constituentPrices,
        'constituentWeights': constituentWeights,
        'nTrades': len(filtered),
        'calculationDate': calculationDate,
        'methodology': str(methodology),
    }


def auditTrail(calculationResults, indexSpec):
    """Create an IOSCO-aligned audit record for an index calculation.

    Parameters
    ----------
    calculationResults : dict
        Output of calculateIndex — must contain 'indexValue', 'calculationDate',
        'nTrades', 'constituentValues', 'constituentWeights'.
    indexSpec : dict
        IndexSpec dict (from schema.py).

    Returns
    -------
    dict
        Keys:

        - 'timestamp' : str — ISO-like timestamp string (YYYY-MM-DD format using
          numpy datetime64 today).
        - 'indexName' : str
        - 'version' : str
        - 'calculationDate' : from calculationResults
        - 'indexValue' : float
        - 'constituentDetail' : list of dicts, one per constituent, with keys
          'constituent', 'vwap', 'weight'.
        - 'dataSourcesUsed' : list — from indexSpec constituents.
        - 'methodologyVersion' : str
        - 'checksum' : int — sum of ASCII codes of str(indexValue)+str(version).

    Notes
    -----
    Checksum is a simple integrity marker: sum of ASCII ordinals of the
    concatenated string of indexValue and methodology version.
    """
    indexValue = calculationResults['indexValue']
    version = indexSpec['version']

    checksum = sum(ord(c) for c in (str(indexValue) + str(version)))

    constituentDetail = [
        {
            'constituent': c,
            'vwap': calculationResults['constituentValues'].get(c, 0.0),
            'weight': calculationResults['constituentWeights'].get(c, 0.0),
        }
        for c in indexSpec['constituents']
    ]

    timestamp = str(np.datetime64('today', 'D'))

    return {
        'timestamp': timestamp,
        'indexName': indexSpec['name'],
        'version': version,
        'calculationDate': calculationResults['calculationDate'],
        'indexValue': indexValue,
        'constituentDetail': constituentDetail,
        'dataSourcesUsed': list(indexSpec['constituents']),
        'methodologyVersion': version,
        'checksum': checksum,
    }


def restatement(originalRecord, correctedValue, reason, analystId):
    """Record a methodology-compliant index restatement.

    Parameters
    ----------
    originalRecord : dict
        The original auditTrail dict (must contain 'indexValue').
    correctedValue : float
        The corrected index value replacing the original.
    reason : str
        Human-readable explanation for the restatement.
    analystId : str
        Identifier of the analyst authorising the restatement.

    Returns
    -------
    dict
        Keys:

        - 'originalValue' : float
        - 'correctedValue' : float
        - 'delta' : float — correctedValue - originalValue.
        - 'reason' : str
        - 'analystId' : str
        - 'timestamp' : str — ISO-like date string.
        - 'restatementId' : int — sequential module-level counter.

    Notes
    -----
    restatementId is a module-level sequential integer that increments with
    each call. It is not persistent across Python sessions.
    """
    _RESTATEMENT_COUNTER[0] += 1

    originalValue = float(originalRecord['indexValue'])
    correctedValue = float(correctedValue)

    return {
        'originalValue': originalValue,
        'correctedValue': correctedValue,
        'delta': correctedValue - originalValue,
        'reason': str(reason),
        'analystId': str(analystId),
        'timestamp': str(np.datetime64('today', 'D')),
        'restatementId': _RESTATEMENT_COUNTER[0],
    }


def rollSchedule(indexSpec, startDate, endDate, step='monthly'):
    """Generate a roll schedule for an index between two dates.

    Parameters
    ----------
    indexSpec : dict
        IndexSpec dict (from schema.py).
    startDate : int
        Start date as an integer ordinal day (or any integer comparable).
    endDate : int
        End date as an integer ordinal day (or any integer comparable).
    step : str, optional
        Roll frequency. One of 'monthly' (default), 'weekly', 'quarterly'.

    Returns
    -------
    dict
        Keys:

        - 'rollDates' : list of int — roll dates as ordinal integers.
        - 'nRolls' : int — number of roll dates generated.
        - 'step' : str — the step parameter used.

    Notes
    -----
    Dates are accepted and returned as integers (ordinal days).
    - 'monthly' : approximately every 30 days.
    - 'weekly'  : approximately every 7 days.
    - 'quarterly': approximately every 91 days.
    The last roll date is the greatest value not exceeding endDate.
    """
    startDate = int(startDate)
    endDate = int(endDate)

    stepMap = {
        'monthly': 30,
        'weekly': 7,
        'quarterly': 91,
    }

    if step not in stepMap:
        raise ValueError(
            f"step must be one of {list(stepMap.keys())}: got '{step}'."
        )

    stepDays = stepMap[step]
    rollDates = []
    current = startDate + stepDays
    while current <= endDate:
        rollDates.append(current)
        current += stepDays

    return {
        'rollDates': rollDates,
        'nRolls': len(rollDates),
        'step': step,
    }


def proxyRegression(targetSeries, proxySeries, method='ols'):
    """Regression-based proxy construction for missing index values.

    Parameters
    ----------
    targetSeries : array-like of float
        Dependent variable (the index or market series to reconstruct).
    proxySeries : array-like of float
        Independent variable (the proxy series).
    method : str, optional
        Regression method. 'ols' (default) for ordinary least squares;
        'huber' for iteratively reweighted least squares with Huber-like
        weights (downweights large residuals).

    Returns
    -------
    dict
        Keys:

        - 'coefficients' : np.ndarray — [slope] (shape (1,)).
        - 'intercept' : float
        - 'rSquared' : float — coefficient of determination in [0, 1].
        - 'predictedValues' : np.ndarray
        - 'residuals' : np.ndarray

    Notes
    -----
    OLS closed-form solution. Huber method uses iteratively reweighted
    least squares (IRLS) with Huber threshold delta = 1.345 * MAD.
    """
    y = np.asarray(targetSeries, dtype=float)
    x = np.asarray(proxySeries, dtype=float)

    n = len(y)
    if n != len(x):
        raise ValueError(
            f"targetSeries and proxySeries must have the same length: "
            f"got {n} and {len(x)}."
        )
    if n < 2:
        raise ValueError("At least 2 observations are required for regression.")

    X = np.column_stack([np.ones(n), x])

    if method == 'ols':
        coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        intercept = float(coeffs[0])
        slope = coeffs[1:]

    elif method == 'huber':
        # IRLS with Huber weights
        coeffs_prev, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        for _ in range(50):
            residuals = y - X @ coeffs_prev
            mad = np.median(np.abs(residuals - np.median(residuals)))
            delta_h = 1.345 * mad if mad > 1e-10 else 1.0
            weights = np.where(
                np.abs(residuals) <= delta_h,
                1.0,
                delta_h / np.abs(residuals)
            )
            W = np.diag(weights)
            XtW = X.T @ W
            coeffs_new = np.linalg.solve(XtW @ X + 1e-10 * np.eye(X.shape[1]),
                                         XtW @ y)
            if np.max(np.abs(coeffs_new - coeffs_prev)) < 1e-8:
                break
            coeffs_prev = coeffs_new
        coeffs = coeffs_prev
        intercept = float(coeffs[0])
        slope = coeffs[1:]

    else:
        raise ValueError(f"method must be 'ols' or 'huber': got '{method}'.")

    predicted = X @ coeffs
    residuals = y - predicted
    ssTot = np.sum((y - np.mean(y)) ** 2)
    ssRes = np.sum(residuals ** 2)
    rSquared = float(1.0 - ssRes / ssTot) if ssTot > 1e-12 else 0.0
    rSquared = float(np.clip(rSquared, 0.0, 1.0))

    return {
        'coefficients': slope,
        'intercept': intercept,
        'rSquared': rSquared,
        'predictedValues': predicted,
        'residuals': residuals,
    }


def backtestIndex(tradeRecords, indexSpec, dates):
    """Backtest index values across a list of calculation dates.

    Parameters
    ----------
    tradeRecords : list of dict
        List of TradeRecord dicts (from schema.py).
    indexSpec : dict
        IndexSpec dict (from schema.py).
    dates : list
        Ordered list of calculation dates (any comparable type).

    Returns
    -------
    dict
        Keys:

        - 'dates' : list — the input dates.
        - 'indexValues' : np.ndarray — index value at each date.
        - 'returns' : np.ndarray — period-over-period log returns (length
          len(dates)-1).
        - 'volatility' : float — annualised volatility of log returns (252
          trading days). NaN if fewer than 2 return observations.
        - 'maxDrawdown' : float — maximum drawdown from peak (non-negative).
          0.0 if index never declines.

    Notes
    -----
    Log returns are computed as ln(V_t / V_{t-1}). Zero index values produce
    NaN returns which are excluded from volatility calculation.
    """
    indexValues = np.array([
        calculateIndex(tradeRecords, indexSpec, d)['indexValue']
        for d in dates
    ])

    if len(indexValues) > 1:
        prev = indexValues[:-1]
        curr = indexValues[1:]
        with np.errstate(divide='ignore', invalid='ignore'):
            logRet = np.where(
                (prev > 0) & (curr > 0),
                np.log(curr / prev),
                np.nan
            )
        validRet = logRet[~np.isnan(logRet)]
        volatility = float(np.std(validRet) * np.sqrt(252)) if len(validRet) > 1 else float('nan')
    else:
        logRet = np.array([])
        volatility = float('nan')

    # Max drawdown
    if len(indexValues) > 0 and np.any(indexValues > 0):
        peak = indexValues[0]
        maxDD = 0.0
        for v in indexValues:
            if v > peak:
                peak = v
            if peak > 0:
                dd = (peak - v) / peak
                if dd > maxDD:
                    maxDD = dd
    else:
        maxDD = 0.0

    return {
        'dates': list(dates),
        'indexValues': indexValues,
        'returns': logRet,
        'volatility': volatility,
        'maxDrawdown': float(maxDD),
    }
