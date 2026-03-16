import numpy as np

# ---------------------------------------------------------------------------
# sipQuant — schema.py
# Data contract layer: validated input containers for all sipQuant functions.
# Every pricing, risk, and index function accepts these schema objects.
# No external dependencies beyond numpy.
# ---------------------------------------------------------------------------

_VALID_TYPES = frozenset({
    'PriceSeries', 'SparsePriceSeries', 'TradeRecord',
    'QuoteSheet', 'ForwardCurve', 'OTCPosition', 'IndexSpec'
})

_VALID_DIRECTIONS = frozenset({
    'buy', 'sell', 'pay_fixed', 'receive_fixed', 'long', 'short'
})

_VALID_WEIGHTS_METHODS = frozenset({
    'equal', 'volume', 'liquidity', 'custom'
})


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------

def PriceSeries(dates, values, source, market, grade=None):
    """
    Validated container for a regularly-spaced price time series.

    Parameters
    ----------
    dates : array-like
        Observation dates. Any comparable type (str, datetime, int ordinal).
    values : array-like of float
        Price observations. Must be same length as dates.
    source : str
        Data source identifier (broker, exchange, internal).
    market : str
        Market identifier (e.g. 'alberta_hay_premium').
    grade : str, optional
        Grade specification (e.g. 'premium_bale_14pct_moisture').

    Returns
    -------
    dict
        Schema object with type='PriceSeries'.

    Raises
    ------
    ValueError
        If dates and values have different lengths, or values is empty.
    """
    dates = np.asarray(dates)
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("PriceSeries cannot be empty.")
    if len(dates) != len(values):
        raise ValueError(
            f"dates and values must have the same length: "
            f"got {len(dates)} dates and {len(values)} values."
        )

    return {
        'type': 'PriceSeries',
        'dates': dates,
        'values': values,
        'source': str(source),
        'market': str(market),
        'grade': grade,
        'n': len(values),
    }


def SparsePriceSeries(dates, values, source, market, maxGapDays=None):
    """
    Validated container for an irregularly-spaced price time series.

    Use this for markets with weekly, monthly, or ad-hoc observations.
    Gap metadata is stored for use in sparse interpolation routines.

    Parameters
    ----------
    dates : array-like
        Observation dates. Must be monotonically non-decreasing.
    values : array-like of float
        Price observations. Must be same length as dates.
    source : str
        Data source identifier.
    market : str
        Market identifier.
    maxGapDays : int, optional
        If provided, flags gaps exceeding this threshold in the gaps array.

    Returns
    -------
    dict
        Schema object with type='SparsePriceSeries'.

    Raises
    ------
    ValueError
        If dates and values have different lengths, or values is empty.
    """
    dates = np.asarray(dates)
    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        raise ValueError("SparsePriceSeries cannot be empty.")
    if len(dates) != len(values):
        raise ValueError(
            f"dates and values must have the same length: "
            f"got {len(dates)} dates and {len(values)} values."
        )

    gapFlags = None
    if maxGapDays is not None and len(dates) > 1:
        try:
            diffs = np.diff(dates.astype('datetime64[D]')).astype(int)
            gapFlags = diffs > maxGapDays
        except (TypeError, ValueError):
            gapFlags = None

    return {
        'type': 'SparsePriceSeries',
        'dates': dates,
        'values': values,
        'source': str(source),
        'market': str(market),
        'maxGapDays': maxGapDays,
        'gapFlags': gapFlags,
        'n': len(values),
    }


def TradeRecord(date, price, volume, grade, origin, destination, counterpartyId):
    """
    Validated container for a single physical trade observation.

    Physical trades are the primary data source for thin commodity markets.
    Each trade record feeds directly into index calculation and proxy regression.

    Parameters
    ----------
    date : any
        Trade date.
    price : float
        Traded price. Must be positive.
    volume : float
        Traded volume (tonnes, bales, units). Must be positive.
    grade : str
        Quality specification at time of trade.
    origin : str
        Origin delivery point identifier.
    destination : str
        Destination delivery point identifier.
    counterpartyId : str
        Anonymised counterparty identifier.

    Returns
    -------
    dict
        Schema object with type='TradeRecord'.

    Raises
    ------
    ValueError
        If price or volume are non-positive.
    """
    price = float(price)
    volume = float(volume)

    if price <= 0:
        raise ValueError(f"price must be positive: got {price}.")
    if volume <= 0:
        raise ValueError(f"volume must be positive: got {volume}.")

    return {
        'type': 'TradeRecord',
        'date': date,
        'price': price,
        'volume': volume,
        'grade': str(grade),
        'origin': str(origin),
        'destination': str(destination),
        'counterpartyId': str(counterpartyId),
    }


def QuoteSheet(date, bid, ask, mid, source, market, grade, tenor):
    """
    Validated OTC quote observation with full metadata.

    Standardises how broker quotes enter the pricing stack. Mid price is
    the primary mark-to-model input when no physical trade has occurred.

    Parameters
    ----------
    date : any
        Quote date.
    bid : float or None
        Bid price. None if not available.
    ask : float or None
        Ask price. None if not available.
    mid : float
        Mid price. Required.
    source : str
        Quote source (broker name or internal).
    market : str
        Market identifier.
    grade : str
        Grade specification.
    tenor : float
        Contract tenor in years (e.g. 0.25 for 3-month).

    Returns
    -------
    dict
        Schema object with type='QuoteSheet'.

    Raises
    ------
    ValueError
        If bid > ask when both are provided, or tenor is negative.
    """
    mid = float(mid)
    tenor = float(tenor)

    if bid is not None and ask is not None:
        bid = float(bid)
        ask = float(ask)
        if bid > ask:
            raise ValueError(
                f"bid ({bid}) cannot exceed ask ({ask})."
            )
    else:
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None

    if tenor < 0:
        raise ValueError(f"tenor must be non-negative: got {tenor}.")

    return {
        'type': 'QuoteSheet',
        'date': date,
        'bid': bid,
        'ask': ask,
        'mid': mid,
        'source': str(source),
        'market': str(market),
        'grade': str(grade),
        'tenor': tenor,
    }


def ForwardCurve(tenors, prices, baseDate, market, methodology='linear'):
    """
    Validated forward curve container.

    Forward curves are the primary input to OTC pricing functions.
    Tenors must be monotonically non-decreasing and non-negative.

    Parameters
    ----------
    tenors : array-like of float
        Contract tenors in years (e.g. [0.25, 0.5, 1.0]).
    prices : array-like of float
        Forward prices at each tenor. Must be positive.
    baseDate : any
        Curve base date (the 'as-of' date).
    market : str
        Market identifier.
    methodology : str, optional
        Interpolation methodology applied to this curve. Default 'linear'.

    Returns
    -------
    dict
        Schema object with type='ForwardCurve'.

    Raises
    ------
    ValueError
        If tenors/prices have different lengths, tenors are not monotone,
        or any price is non-positive.
    """
    tenors = np.asarray(tenors, dtype=float)
    prices = np.asarray(prices, dtype=float)

    if len(tenors) == 0:
        raise ValueError("ForwardCurve cannot be empty.")
    if len(tenors) != len(prices):
        raise ValueError(
            f"tenors and prices must have the same length: "
            f"got {len(tenors)} tenors and {len(prices)} prices."
        )
    if np.any(tenors < 0):
        raise ValueError("All tenors must be non-negative.")
    if not np.all(np.diff(tenors) >= 0):
        raise ValueError("tenors must be monotonically non-decreasing.")
    if np.any(prices <= 0):
        raise ValueError("All forward prices must be positive.")

    return {
        'type': 'ForwardCurve',
        'tenors': tenors,
        'prices': prices,
        'baseDate': baseDate,
        'market': str(market),
        'methodology': str(methodology),
        'n': len(tenors),
    }


def OTCPosition(instrumentType, direction, notional, strikeOrFixed,
                expiry, counterpartyId, greeks=None):
    """
    Validated OTC position record for book aggregation.

    Standardised position structure ensures book.py aggregation works
    regardless of instrument type. Greeks default to zero if not provided.

    Parameters
    ----------
    instrumentType : str
        Instrument type (e.g. 'commodity_swap', 'collar', 'physical_forward').
    direction : str
        Trade direction. One of: 'buy', 'sell', 'pay_fixed', 'receive_fixed',
        'long', 'short'.
    notional : float
        Position notional (tonnes, bales, MWh, etc.). Must be positive.
    strikeOrFixed : float
        Strike price (options) or fixed rate (swaps).
    expiry : any
        Contract expiry date.
    counterpartyId : str
        Counterparty identifier.
    greeks : dict, optional
        Greeks dict with keys: delta, gamma, vega, theta, rho.
        Defaults to all zeros.

    Returns
    -------
    dict
        Schema object with type='OTCPosition'.

    Raises
    ------
    ValueError
        If direction is invalid or notional is non-positive.
    """
    direction = str(direction).lower()
    if direction not in _VALID_DIRECTIONS:
        raise ValueError(
            f"direction must be one of {sorted(_VALID_DIRECTIONS)}: got '{direction}'."
        )

    notional = float(notional)
    if notional <= 0:
        raise ValueError(f"notional must be positive: got {notional}.")

    if greeks is None:
        greeks = {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0, 'rho': 0.0}
    else:
        greeks = {
            'delta': float(greeks.get('delta', 0.0)),
            'gamma': float(greeks.get('gamma', 0.0)),
            'vega': float(greeks.get('vega', 0.0)),
            'theta': float(greeks.get('theta', 0.0)),
            'rho': float(greeks.get('rho', 0.0)),
        }

    return {
        'type': 'OTCPosition',
        'instrumentType': str(instrumentType),
        'direction': direction,
        'notional': notional,
        'strikeOrFixed': float(strikeOrFixed),
        'expiry': expiry,
        'counterpartyId': str(counterpartyId),
        'greeks': greeks,
    }


def IndexSpec(name, version, constituents, weightsMethod, rollRule, effectiveDate):
    """
    Immutable index methodology specification.

    The governing document for OTC settlement. Records the complete
    methodology version that will be pinned to each calculation date.
    Immutability ensures no retroactive changes are possible.

    Parameters
    ----------
    name : str
        Index name (e.g. 'SIP-AHI-001').
    version : str
        Methodology version string (e.g. '1.0').
    constituents : list of str
        Market or grade identifiers that contribute to the index.
    weightsMethod : str
        Constituent weighting method. One of: 'equal', 'volume',
        'liquidity', 'custom'.
    rollRule : str
        Roll logic description (e.g. 'monthly_last_business_day').
    effectiveDate : any
        Date from which this methodology version is effective.

    Returns
    -------
    dict
        Schema object with type='IndexSpec'.

    Raises
    ------
    ValueError
        If weightsMethod is not a recognised value, or constituents is empty.
    """
    weightsMethod = str(weightsMethod).lower()
    if weightsMethod not in _VALID_WEIGHTS_METHODS:
        raise ValueError(
            f"weightsMethod must be one of {sorted(_VALID_WEIGHTS_METHODS)}: "
            f"got '{weightsMethod}'."
        )

    constituents = list(constituents)
    if len(constituents) == 0:
        raise ValueError("IndexSpec must have at least one constituent.")

    return {
        'type': 'IndexSpec',
        'name': str(name),
        'version': str(version),
        'constituents': constituents,
        'weightsMethod': weightsMethod,
        'rollRule': str(rollRule),
        'effectiveDate': effectiveDate,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(obj):
    """
    Universal validation dispatcher.

    Runs type-specific validation checks on any sipQuant schema object
    and returns a list of error strings. An empty list means the object
    is valid.

    Parameters
    ----------
    obj : dict
        Any sipQuant schema object returned by a constructor in this module.

    Returns
    -------
    list of str
        Validation errors. Empty list if object is valid.

    Examples
    --------
    >>> ps = PriceSeries([1, 2, 3], [10.0, 11.0, 12.0], 'broker_a', 'hay')
    >>> validate(ps)
    []
    """
    errors = []

    if not isinstance(obj, dict):
        return ['Object must be a dict.']

    objType = obj.get('type')
    if objType is None:
        return ["Object missing 'type' field."]
    if objType not in _VALID_TYPES:
        return [f"Unknown type '{objType}'. Valid types: {sorted(_VALID_TYPES)}."]

    if objType in ('PriceSeries', 'SparsePriceSeries'):
        if 'values' not in obj:
            errors.append("Missing 'values' field.")
        elif len(obj['values']) == 0:
            errors.append("'values' array is empty.")
        if 'dates' not in obj:
            errors.append("Missing 'dates' field.")
        elif 'values' in obj and len(obj['dates']) != len(obj['values']):
            errors.append(
                f"'dates' and 'values' length mismatch: "
                f"{len(obj['dates'])} vs {len(obj['values'])}."
            )
        if 'market' not in obj or not obj['market']:
            errors.append("Missing or empty 'market' field.")

    elif objType == 'TradeRecord':
        price = obj.get('price')
        volume = obj.get('volume')
        if price is None or price <= 0:
            errors.append(f"'price' must be positive: got {price}.")
        if volume is None or volume <= 0:
            errors.append(f"'volume' must be positive: got {volume}.")
        for field in ('grade', 'origin', 'destination', 'counterpartyId'):
            if not obj.get(field):
                errors.append(f"Missing or empty '{field}' field.")

    elif objType == 'QuoteSheet':
        if obj.get('mid') is None:
            errors.append("'mid' is required.")
        bid = obj.get('bid')
        ask = obj.get('ask')
        if bid is not None and ask is not None and bid > ask:
            errors.append(f"bid ({bid}) > ask ({ask}).")
        if obj.get('tenor', -1) < 0:
            errors.append("'tenor' must be non-negative.")

    elif objType == 'ForwardCurve':
        tenors = obj.get('tenors')
        prices = obj.get('prices')
        if tenors is None or prices is None:
            errors.append("ForwardCurve missing 'tenors' or 'prices'.")
        else:
            if len(tenors) == 0:
                errors.append("'tenors' array is empty.")
            if len(tenors) != len(prices):
                errors.append(
                    f"'tenors' and 'prices' length mismatch: "
                    f"{len(tenors)} vs {len(prices)}."
                )
            if np.any(np.asarray(tenors) < 0):
                errors.append("All tenors must be non-negative.")
            if len(tenors) > 1 and not np.all(np.diff(np.asarray(tenors)) >= 0):
                errors.append("'tenors' must be monotonically non-decreasing.")
            if np.any(np.asarray(prices) <= 0):
                errors.append("All forward prices must be positive.")

    elif objType == 'OTCPosition':
        if obj.get('direction') not in _VALID_DIRECTIONS:
            errors.append(
                f"'direction' must be one of {sorted(_VALID_DIRECTIONS)}: "
                f"got '{obj.get('direction')}'."
            )
        if obj.get('notional', 0) <= 0:
            errors.append(f"'notional' must be positive: got {obj.get('notional')}.")
        greeks = obj.get('greeks', {})
        for g in ('delta', 'gamma', 'vega', 'theta', 'rho'):
            if g not in greeks:
                errors.append(f"Greeks dict missing '{g}'.")

    elif objType == 'IndexSpec':
        if obj.get('weightsMethod') not in _VALID_WEIGHTS_METHODS:
            errors.append(
                f"'weightsMethod' must be one of {sorted(_VALID_WEIGHTS_METHODS)}."
            )
        if not obj.get('constituents'):
            errors.append("'constituents' must be a non-empty list.")
        if not obj.get('name'):
            errors.append("'name' is required.")
        if not obj.get('version'):
            errors.append("'version' is required.")

    return errors
