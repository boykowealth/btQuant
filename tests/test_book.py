import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import sipQuant as sq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_pos(delta=0.5, notional=100.0, direction='buy', instrument='commodity_swap'):
    """Create a test OTCPosition dict."""
    return sq.schema.OTCPosition(
        instrument,
        direction,
        notional,
        100.0,
        '2026-12-31',
        'CP001',
        greeks={
            'delta': delta,
            'gamma': 0.01,
            'vega': 5.0,
            'theta': -0.1,
            'rho': 0.2,
        },
    )


# ---------------------------------------------------------------------------
# netGreeks
# ---------------------------------------------------------------------------

def test_netGreeks_sums_delta():
    """Two buy positions with the same delta sum to double the contribution."""
    pos1 = make_pos(delta=0.5, notional=100.0)
    pos2 = make_pos(delta=0.5, notional=100.0)
    result = sq.book.netGreeks([pos1, pos2])
    assert isinstance(result, dict), "netGreeks must return a dict"
    expected_delta = 0.5 * 100.0 * 1 + 0.5 * 100.0 * 1
    assert abs(result['delta'] - expected_delta) < 1e-9, (
        f"Expected delta={expected_delta}, got {result['delta']}"
    )


def test_netGreeks_opposite_directions_cancel():
    """A buy and a sell with identical Greeks should net to zero delta."""
    pos_buy = make_pos(delta=0.5, notional=100.0, direction='buy')
    pos_sell = make_pos(delta=0.5, notional=100.0, direction='sell')
    result = sq.book.netGreeks([pos_buy, pos_sell])
    assert abs(result['delta']) < 1e-9, (
        f"Expected net delta ~0, got {result['delta']}"
    )


def test_netGreeks_returns_all_keys():
    """Result must contain all five Greek keys."""
    pos = make_pos()
    result = sq.book.netGreeks([pos])
    for key in ('delta', 'gamma', 'vega', 'theta', 'rho'):
        assert key in result, f"Missing key '{key}' in netGreeks result"


def test_netGreeks_empty_book():
    """Empty position list must return zero Greeks."""
    result = sq.book.netGreeks([])
    for key in ('delta', 'gamma', 'vega', 'theta', 'rho'):
        assert result[key] == 0.0, f"Expected 0 for '{key}' on empty book"


# ---------------------------------------------------------------------------
# hedgeRatios
# ---------------------------------------------------------------------------

def test_hedgeRatios_basic():
    """hedgeUnits ≈ -netDelta / hedgeInstrumentDelta."""
    pos = make_pos(delta=0.5, notional=200.0, direction='buy')
    result = sq.book.hedgeRatios([pos], hedgeInstrumentDelta=1.0)
    assert isinstance(result, dict), "hedgeRatios must return a dict"
    net_delta = result['netDelta']
    expected_units = -net_delta / 1.0
    assert abs(result['hedgeUnits'] - expected_units) < 1e-9, (
        f"Expected hedgeUnits={expected_units}, got {result['hedgeUnits']}"
    )


def test_hedgeRatios_keys_present():
    """Result must contain hedgeUnits, netDelta, residualDelta."""
    pos = make_pos()
    result = sq.book.hedgeRatios([pos], hedgeInstrumentDelta=0.5)
    for key in ('hedgeUnits', 'netDelta', 'residualDelta'):
        assert key in result, f"Missing key '{key}' in hedgeRatios result"


def test_hedgeRatios_residual_is_small():
    """After rounding hedgeUnits, residualDelta should be less than one
    hedgeInstrumentDelta in magnitude."""
    pos = make_pos(delta=0.5, notional=150.0)
    hedge_delta = 1.0
    result = sq.book.hedgeRatios([pos], hedgeInstrumentDelta=hedge_delta)
    assert abs(result['residualDelta']) <= abs(hedge_delta * 0.5 + 1e-9), (
        f"residualDelta {result['residualDelta']} unexpectedly large"
    )


# ---------------------------------------------------------------------------
# pnlAttribution
# ---------------------------------------------------------------------------

def test_pnlAttribution_delta_component():
    """Delta P&L = netDelta * dS (no gamma, vega, theta)."""
    pos = make_pos(delta=0.5, notional=100.0, direction='buy')
    dS = 2.0
    result = sq.book.pnlAttribution([pos], priceMoves={'default': dS})
    assert isinstance(result, dict), "pnlAttribution must return a dict"
    expected_delta_comp = 0.5 * 100.0 * dS
    assert abs(result['deltaComponent'] - expected_delta_comp) < 1e-9, (
        f"Expected deltaComponent={expected_delta_comp}, got {result['deltaComponent']}"
    )


def test_pnlAttribution_total_pnl_consistent():
    """totalPnL = sum of all components."""
    pos = make_pos(delta=0.5, notional=100.0)
    result = sq.book.pnlAttribution(
        [pos],
        priceMoves={'default': 1.0},
        volMoves={'default': 0.02},
        timeDecay=1.0,
    )
    component_sum = (
        result['deltaComponent']
        + result['gammaComponent']
        + result['vegaComponent']
        + result['thetaComponent']
    )
    assert abs(result['totalPnL'] - component_sum) < 1e-9, (
        f"totalPnL {result['totalPnL']} != sum of components {component_sum}"
    )


def test_pnlAttribution_keys():
    """All expected keys must be present in the result."""
    pos = make_pos()
    result = sq.book.pnlAttribution([pos], priceMoves={'default': 1.0})
    for key in ('totalPnL', 'deltaComponent', 'gammaComponent',
                'vegaComponent', 'thetaComponent', 'totalByPosition'):
        assert key in result, f"Missing key '{key}' in pnlAttribution result"


def test_pnlAttribution_totalByPosition_shape():
    """totalByPosition must have the same length as positions."""
    positions = [make_pos(), make_pos(delta=0.3)]
    result = sq.book.pnlAttribution(positions, priceMoves={'default': 1.0})
    assert len(result['totalByPosition']) == 2, (
        f"Expected length 2, got {len(result['totalByPosition'])}"
    )


# ---------------------------------------------------------------------------
# scenarioShock
# ---------------------------------------------------------------------------

def test_scenarioShock_returns_list():
    """scenarioResults must be a list with one entry per scenario."""
    positions = [make_pos(), make_pos(delta=0.3)]
    scenarios = [
        {'name': 'crash', 'priceShock': -10.0, 'volShock': 0.05},
        {'name': 'rally', 'priceShock': 10.0, 'volShock': -0.02},
    ]
    result = sq.book.scenarioShock(positions, scenarios)
    assert isinstance(result, dict), "scenarioShock must return a dict"
    assert 'scenarioResults' in result, "Missing 'scenarioResults' key"
    assert len(result['scenarioResults']) == 2, (
        f"Expected 2 scenario results, got {len(result['scenarioResults'])}"
    )


def test_scenarioShock_result_keys():
    """Each scenario result must contain name, pnl, deltaContrib, vegaContrib."""
    pos = make_pos()
    result = sq.book.scenarioShock(
        [pos],
        [{'name': 'test', 'priceShock': 1.0, 'volShock': 0.01}],
    )
    sr = result['scenarioResults'][0]
    for key in ('name', 'pnl', 'deltaContrib', 'vegaContrib'):
        assert key in sr, f"Missing key '{key}' in scenario result"


def test_scenarioShock_negative_price_shock():
    """A price drop on a long book should produce a negative P&L."""
    pos = make_pos(delta=0.5, notional=100.0, direction='buy')
    result = sq.book.scenarioShock(
        [pos],
        [{'name': 'down', 'priceShock': -5.0, 'volShock': 0.0}],
    )
    pnl = result['scenarioResults'][0]['pnl']
    # Delta contribution: 0.5 * 100 * (-5) = -250; gamma: 0.5*0.01*100*25 = 12.5
    assert pnl < 0, f"Expected negative P&L for price drop, got {pnl}"


# ---------------------------------------------------------------------------
# bookSummary
# ---------------------------------------------------------------------------

def test_bookSummary_total_notional():
    """totalNotional must equal sum of all position notionals."""
    p1 = make_pos(notional=100.0)
    p2 = make_pos(notional=250.0)
    result = sq.book.bookSummary([p1, p2])
    assert isinstance(result, dict), "bookSummary must return a dict"
    assert abs(result['totalNotional'] - 350.0) < 1e-9, (
        f"Expected totalNotional=350, got {result['totalNotional']}"
    )


def test_bookSummary_nPositions():
    """nPositions must equal the length of the input list."""
    positions = [make_pos() for _ in range(5)]
    result = sq.book.bookSummary(positions)
    assert result['nPositions'] == 5, (
        f"Expected nPositions=5, got {result['nPositions']}"
    )


def test_bookSummary_keys():
    """All expected keys must be present."""
    result = sq.book.bookSummary([make_pos()])
    for key in ('totalNotional', 'nPositions', 'byInstrument',
                'netGreeks', 'concentrationRisk'):
        assert key in result, f"Missing key '{key}' in bookSummary result"


def test_bookSummary_single_position_concentration():
    """A single position should have concentrationRisk = 1.0."""
    result = sq.book.bookSummary([make_pos(notional=500.0)])
    assert abs(result['concentrationRisk'] - 1.0) < 1e-9, (
        f"Expected concentrationRisk=1.0, got {result['concentrationRisk']}"
    )


def test_bookSummary_by_instrument():
    """byInstrument must aggregate notional by instrumentType."""
    p1 = make_pos(notional=100.0, instrument='commodity_swap')
    p2 = make_pos(notional=200.0, instrument='collar')
    result = sq.book.bookSummary([p1, p2])
    assert 'commodity_swap' in result['byInstrument'], "Missing 'commodity_swap' key"
    assert 'collar' in result['byInstrument'], "Missing 'collar' key"
    assert abs(result['byInstrument']['commodity_swap'] - 100.0) < 1e-9
    assert abs(result['byInstrument']['collar'] - 200.0) < 1e-9


# ---------------------------------------------------------------------------
# marginEstimate
# ---------------------------------------------------------------------------

def test_marginEstimate_positive():
    """totalMargin must be strictly positive for a non-trivial position."""
    pos = make_pos(delta=0.5, notional=1000.0)
    result = sq.book.marginEstimate([pos])
    assert isinstance(result, dict), "marginEstimate must return a dict"
    assert result['totalMargin'] > 0, (
        f"Expected positive totalMargin, got {result['totalMargin']}"
    )


def test_marginEstimate_keys():
    """Result must contain initialMargin, variationMargin, totalMargin."""
    result = sq.book.marginEstimate([make_pos()])
    for key in ('initialMargin', 'variationMargin', 'totalMargin'):
        assert key in result, f"Missing key '{key}' in marginEstimate result"


def test_marginEstimate_total_is_sum():
    """totalMargin == initialMargin + variationMargin."""
    result = sq.book.marginEstimate([make_pos(delta=0.6, notional=500.0)])
    assert abs(result['totalMargin']
               - (result['initialMargin'] + result['variationMargin'])) < 1e-9, (
        "totalMargin must equal initialMargin + variationMargin"
    )


def test_marginEstimate_custom_rates():
    """Doubling initialMarginRate should approximately double initialMargin."""
    pos = make_pos(delta=0.5, notional=100.0)
    r1 = sq.book.marginEstimate([pos], initialMarginRate=0.1)
    r2 = sq.book.marginEstimate([pos], initialMarginRate=0.2)
    assert abs(r2['initialMargin'] - 2 * r1['initialMargin']) < 1e-9, (
        "initialMargin should scale linearly with initialMarginRate"
    )


if __name__ == '__main__':
    # Simple runner for manual execution without pytest
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
