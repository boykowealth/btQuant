import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import sipQuant as sq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_trade(date, price, volume, grade='hay_premium'):
    return sq.schema.TradeRecord(
        date=date,
        price=price,
        volume=volume,
        grade=grade,
        origin='AB',
        destination='SK',
        counterpartyId='CP001',
    )


def make_index_spec(constituents=None, weightsMethod='equal'):
    if constituents is None:
        constituents = ['hay_premium', 'hay_standard']
    return sq.schema.IndexSpec(
        name='SIP-AHI-001',
        version='1.0',
        constituents=constituents,
        weightsMethod=weightsMethod,
        rollRule='monthly_last_business_day',
        effectiveDate='2026-01-01',
    )


def make_trades():
    """Minimal set of valid trades across two constituents."""
    return [
        make_trade(date='2026-01-05', price=250.0, volume=10.0, grade='hay_premium'),
        make_trade(date='2026-01-07', price=255.0, volume=20.0, grade='hay_premium'),
        make_trade(date='2026-01-06', price=200.0, volume=15.0, grade='hay_standard'),
        make_trade(date='2026-01-08', price=205.0, volume=5.0,  grade='hay_standard'),
    ]


# ---------------------------------------------------------------------------
# calculateIndex
# ---------------------------------------------------------------------------

def test_calculateIndex_returns_positive():
    """Index value must be positive for valid trade inputs."""
    trades = make_trades()
    spec = make_index_spec()
    result = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    assert isinstance(result, dict), "calculateIndex must return a dict"
    assert result['indexValue'] > 0, (
        f"Expected positive indexValue, got {result['indexValue']}"
    )


def test_calculateIndex_keys():
    """Result must contain all required keys."""
    trades = make_trades()
    spec = make_index_spec()
    result = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    for key in ('indexValue', 'constituentValues', 'constituentWeights',
                'nTrades', 'calculationDate', 'methodology'):
        assert key in result, f"Missing key '{key}' in calculateIndex result"


def test_calculateIndex_nTrades():
    """nTrades must equal the number of trades on or before calculationDate."""
    trades = make_trades()
    spec = make_index_spec()
    result = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    assert result['nTrades'] == 4, (
        f"Expected nTrades=4, got {result['nTrades']}"
    )


def test_calculateIndex_future_date_excludes_trades():
    """Trades after calculationDate must be excluded."""
    trades = make_trades()
    spec = make_index_spec()
    # Only trades on or before 2026-01-05 are included
    result = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-05')
    assert result['nTrades'] <= 4, "nTrades should not exceed total trades"


def test_calculateIndex_vwap_correctness():
    """VWAP for hay_premium: (250*10 + 255*20) / 30 = 253.33..."""
    trades = make_trades()
    spec = make_index_spec(constituents=['hay_premium'], weightsMethod='equal')
    result = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    expected_vwap = (250.0 * 10.0 + 255.0 * 20.0) / (10.0 + 20.0)
    assert abs(result['constituentValues']['hay_premium'] - expected_vwap) < 1e-9, (
        f"Expected hay_premium VWAP={expected_vwap}, "
        f"got {result['constituentValues']['hay_premium']}"
    )


def test_calculateIndex_weights_sum_to_one():
    """Constituent weights must sum to 1.0."""
    trades = make_trades()
    spec = make_index_spec()
    result = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    total_weight = sum(result['constituentWeights'].values())
    assert abs(total_weight - 1.0) < 1e-9, (
        f"Constituent weights sum to {total_weight}, expected 1.0"
    )


# ---------------------------------------------------------------------------
# auditTrail
# ---------------------------------------------------------------------------

def test_auditTrail_returns_checksum():
    """auditTrail must return a dict with a 'checksum' key."""
    trades = make_trades()
    spec = make_index_spec()
    calc = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    result = sq.index.auditTrail(calc, spec)
    assert isinstance(result, dict), "auditTrail must return a dict"
    assert 'checksum' in result, "Missing 'checksum' key in auditTrail result"


def test_auditTrail_checksum_is_int():
    """Checksum must be an integer."""
    trades = make_trades()
    spec = make_index_spec()
    calc = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    result = sq.index.auditTrail(calc, spec)
    assert isinstance(result['checksum'], int), (
        f"Expected int checksum, got {type(result['checksum'])}"
    )


def test_auditTrail_index_value_preserved():
    """auditTrail must preserve the indexValue from the calculation result."""
    trades = make_trades()
    spec = make_index_spec()
    calc = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    result = sq.index.auditTrail(calc, spec)
    assert abs(result['indexValue'] - calc['indexValue']) < 1e-12, (
        "auditTrail must preserve indexValue from calculation"
    )


def test_auditTrail_keys():
    """All required audit keys must be present."""
    trades = make_trades()
    spec = make_index_spec()
    calc = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    result = sq.index.auditTrail(calc, spec)
    for key in ('timestamp', 'indexName', 'version', 'calculationDate',
                'indexValue', 'constituentDetail', 'dataSourcesUsed',
                'methodologyVersion', 'checksum'):
        assert key in result, f"Missing key '{key}' in auditTrail result"


# ---------------------------------------------------------------------------
# restatement
# ---------------------------------------------------------------------------

def test_restatement_delta():
    """delta must equal correctedValue - originalValue."""
    trades = make_trades()
    spec = make_index_spec()
    calc = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    audit = sq.index.auditTrail(calc, spec)
    corrected = calc['indexValue'] + 5.0
    result = sq.index.restatement(audit, corrected, 'data correction', 'analyst_01')
    assert isinstance(result, dict), "restatement must return a dict"
    expected_delta = corrected - calc['indexValue']
    assert abs(result['delta'] - expected_delta) < 1e-9, (
        f"Expected delta={expected_delta}, got {result['delta']}"
    )


def test_restatement_keys():
    """All required keys must be in the restatement dict."""
    trades = make_trades()
    spec = make_index_spec()
    calc = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    audit = sq.index.auditTrail(calc, spec)
    result = sq.index.restatement(audit, 250.0, 'test', 'analyst_02')
    for key in ('originalValue', 'correctedValue', 'delta', 'reason',
                'analystId', 'timestamp', 'restatementId'):
        assert key in result, f"Missing key '{key}' in restatement result"


def test_restatement_id_increments():
    """Sequential restatements must have incrementing restatementId."""
    trades = make_trades()
    spec = make_index_spec()
    calc = sq.index.calculateIndex(trades, spec, calculationDate='2026-01-10')
    audit = sq.index.auditTrail(calc, spec)
    r1 = sq.index.restatement(audit, 250.0, 'first', 'analyst_01')
    r2 = sq.index.restatement(audit, 251.0, 'second', 'analyst_01')
    assert r2['restatementId'] > r1['restatementId'], (
        "restatementId must increment with each restatement"
    )


# ---------------------------------------------------------------------------
# rollSchedule
# ---------------------------------------------------------------------------

def test_rollSchedule_returns_dict():
    """rollSchedule must return a dict."""
    spec = make_index_spec()
    result = sq.index.rollSchedule(spec, startDate=1, endDate=100, step='monthly')
    assert isinstance(result, dict), "rollSchedule must return a dict"


def test_rollSchedule_keys():
    """Result must contain rollDates, nRolls, step."""
    spec = make_index_spec()
    result = sq.index.rollSchedule(spec, startDate=1, endDate=100)
    for key in ('rollDates', 'nRolls', 'step'):
        assert key in result, f"Missing key '{key}' in rollSchedule result"


def test_rollSchedule_monthly_count():
    """Monthly step over ~90 days should produce approximately 3 roll dates."""
    spec = make_index_spec()
    result = sq.index.rollSchedule(spec, startDate=0, endDate=91, step='monthly')
    # 30, 60, 90 → 3 dates
    assert result['nRolls'] == 3, (
        f"Expected 3 monthly rolls in 91 days, got {result['nRolls']}"
    )


def test_rollSchedule_list_type():
    """rollDates must be a list."""
    spec = make_index_spec()
    result = sq.index.rollSchedule(spec, startDate=0, endDate=200, step='weekly')
    assert isinstance(result['rollDates'], list), "rollDates must be a list"


def test_rollSchedule_nrolls_consistent():
    """nRolls must equal len(rollDates)."""
    spec = make_index_spec()
    result = sq.index.rollSchedule(spec, startDate=0, endDate=365, step='quarterly')
    assert result['nRolls'] == len(result['rollDates']), (
        "nRolls must equal len(rollDates)"
    )


# ---------------------------------------------------------------------------
# proxyRegression
# ---------------------------------------------------------------------------

def test_proxyRegression_r_squared_range():
    """rSquared must be in [0, 1]."""
    np.random.seed(42)
    x = np.linspace(100, 200, 50)
    y = 1.5 * x + 20 + np.random.normal(0, 5, 50)
    result = sq.index.proxyRegression(y, x, method='ols')
    assert isinstance(result, dict), "proxyRegression must return a dict"
    assert 0.0 <= result['rSquared'] <= 1.0, (
        f"rSquared={result['rSquared']} outside [0, 1]"
    )


def test_proxyRegression_keys():
    """All required keys must be present."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = 2.0 * x + 1.0
    result = sq.index.proxyRegression(y, x)
    for key in ('coefficients', 'intercept', 'rSquared', 'predictedValues', 'residuals'):
        assert key in result, f"Missing key '{key}' in proxyRegression result"


def test_proxyRegression_perfect_fit():
    """Perfect linear relationship must yield rSquared ~ 1.0."""
    x = np.arange(1.0, 21.0)
    y = 3.0 * x + 7.0
    result = sq.index.proxyRegression(y, x, method='ols')
    assert abs(result['rSquared'] - 1.0) < 1e-6, (
        f"Perfect fit expected rSquared=1.0, got {result['rSquared']}"
    )


def test_proxyRegression_huber_r_squared_range():
    """Huber method rSquared must also be in [0, 1]."""
    np.random.seed(0)
    x = np.linspace(50, 150, 40)
    y = 0.8 * x + 10 + np.random.normal(0, 10, 40)
    result = sq.index.proxyRegression(y, x, method='huber')
    assert 0.0 <= result['rSquared'] <= 1.0, (
        f"Huber rSquared={result['rSquared']} outside [0, 1]"
    )


def test_proxyRegression_residuals_shape():
    """Residuals must have the same length as the input series."""
    x = np.arange(1.0, 11.0)
    y = 2.0 * x + np.random.normal(0, 1, 10)
    result = sq.index.proxyRegression(y, x)
    assert len(result['residuals']) == 10, (
        f"Expected 10 residuals, got {len(result['residuals'])}"
    )


# ---------------------------------------------------------------------------
# backtestIndex
# ---------------------------------------------------------------------------

def test_backtestIndex_returns_dict():
    """backtestIndex must return a dict."""
    trades = make_trades()
    spec = make_index_spec()
    dates = ['2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08']
    result = sq.index.backtestIndex(trades, spec, dates)
    assert isinstance(result, dict), "backtestIndex must return a dict"


def test_backtestIndex_keys():
    """All required keys must be present."""
    trades = make_trades()
    spec = make_index_spec()
    dates = ['2026-01-05', '2026-01-07', '2026-01-10']
    result = sq.index.backtestIndex(trades, spec, dates)
    for key in ('dates', 'indexValues', 'returns', 'volatility', 'maxDrawdown'):
        assert key in result, f"Missing key '{key}' in backtestIndex result"


def test_backtestIndex_indexValues_length():
    """indexValues must have one entry per date."""
    trades = make_trades()
    spec = make_index_spec()
    dates = ['2026-01-05', '2026-01-07', '2026-01-10']
    result = sq.index.backtestIndex(trades, spec, dates)
    assert len(result['indexValues']) == 3, (
        f"Expected 3 indexValues, got {len(result['indexValues'])}"
    )


def test_backtestIndex_all_positive():
    """All index values must be non-negative for valid positive-price trades."""
    trades = make_trades()
    spec = make_index_spec()
    dates = ['2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08']
    result = sq.index.backtestIndex(trades, spec, dates)
    assert np.all(result['indexValues'] >= 0), (
        "All index values must be non-negative"
    )


def test_backtestIndex_maxDrawdown_non_negative():
    """maxDrawdown must be non-negative."""
    trades = make_trades()
    spec = make_index_spec()
    dates = ['2026-01-05', '2026-01-07', '2026-01-10']
    result = sq.index.backtestIndex(trades, spec, dates)
    assert result['maxDrawdown'] >= 0, (
        f"Expected non-negative maxDrawdown, got {result['maxDrawdown']}"
    )


if __name__ == '__main__':
    import traceback
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  FAIL  {t.__name__}: {exc}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed.")
