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


def sample_returns(n=252, seed=42):
    np.random.seed(seed)
    return np.random.normal(0.0, 0.02, n)


def sample_volumes(n=252, seed=0):
    np.random.seed(seed)
    return np.abs(np.random.normal(1000.0, 200.0, n)) + 100.0


# ---------------------------------------------------------------------------
# liquidityAdjustedVar
# ---------------------------------------------------------------------------

def test_lvar_geq_var():
    """LVAR must be greater than or equal to standard VaR."""
    returns = sample_returns()
    volumes = sample_volumes()
    result = sq.liquidity.liquidityAdjustedVar(returns, volumes, alpha=0.05)
    assert isinstance(result, dict), "liquidityAdjustedVar must return a dict"
    assert result['lvar'] >= result['var'], (
        f"Expected lvar >= var, got lvar={result['lvar']}, var={result['var']}"
    )


def test_lvar_keys():
    """Result must contain lvar, var, liquidityCost, spreadCost."""
    returns = sample_returns()
    volumes = sample_volumes()
    result = sq.liquidity.liquidityAdjustedVar(returns, volumes)
    for key in ('lvar', 'var', 'liquidityCost', 'spreadCost'):
        assert key in result, f"Missing key '{key}' in liquidityAdjustedVar result"


def test_lvar_explicit_spread():
    """Providing explicit spreadCost must be reflected in the result."""
    returns = sample_returns()
    volumes = sample_volumes()
    result = sq.liquidity.liquidityAdjustedVar(returns, volumes, spreadCost=0.01)
    assert abs(result['spreadCost'] - 0.01) < 1e-12, (
        f"Expected spreadCost=0.01, got {result['spreadCost']}"
    )
    assert abs(result['liquidityCost'] - 0.005) < 1e-12, (
        f"Expected liquidityCost=0.005, got {result['liquidityCost']}"
    )


def test_lvar_liquid_market_tighter_spread():
    """Higher average volume should produce a tighter estimated spread."""
    returns = sample_returns()
    low_vol = np.full(252, 10.0)
    high_vol = np.full(252, 10000.0)
    r_low = sq.liquidity.liquidityAdjustedVar(returns, low_vol)
    r_high = sq.liquidity.liquidityAdjustedVar(returns, high_vol)
    assert r_high['spreadCost'] <= r_low['spreadCost'], (
        "Higher volume should produce lower or equal spread cost"
    )


# ---------------------------------------------------------------------------
# marketImpact
# ---------------------------------------------------------------------------

def test_marketImpact_positive():
    """totalImpact must be positive for a non-zero trade."""
    result = sq.liquidity.marketImpact(tradeSize=1000, adv=10000)
    assert isinstance(result, dict), "marketImpact must return a dict"
    assert result['totalImpact'] > 0, (
        f"Expected positive totalImpact, got {result['totalImpact']}"
    )


def test_marketImpact_keys():
    """All required keys must be present."""
    result = sq.liquidity.marketImpact(tradeSize=500, adv=5000)
    for key in ('permanentImpact', 'temporaryImpact', 'totalImpact', 'impactBps'):
        assert key in result, f"Missing key '{key}' in marketImpact result"


def test_marketImpact_linear_model():
    """Linear model: permanent = gamma*(x/adv), temporary = eta*(x/adv)."""
    x, adv, eta, gamma = 1000.0, 10000.0, 0.1, 0.1
    result = sq.liquidity.marketImpact(x, adv, model='linear', eta=eta, gamma=gamma)
    expected_perm = gamma * x / adv
    expected_temp = eta * x / adv
    assert abs(result['permanentImpact'] - expected_perm) < 1e-12
    assert abs(result['temporaryImpact'] - expected_temp) < 1e-12


def test_marketImpact_almgren_chriss_model():
    """AC model: permanentImpact = gamma*(x/adv), temporaryImpact = eta*sqrt(x/adv)."""
    x, adv, eta, gamma = 1000.0, 10000.0, 0.1, 0.1
    result = sq.liquidity.marketImpact(x, adv, model='almgren-chriss',
                                       eta=eta, gamma=gamma)
    expected_perm = gamma * x / adv
    expected_temp = eta * np.sqrt(x / adv)
    assert abs(result['permanentImpact'] - expected_perm) < 1e-12
    assert abs(result['temporaryImpact'] - expected_temp) < 1e-12


def test_marketImpact_impactBps():
    """impactBps must equal totalImpact * 10000."""
    result = sq.liquidity.marketImpact(1000, 10000)
    assert abs(result['impactBps'] - result['totalImpact'] * 10000) < 1e-9, (
        "impactBps must equal totalImpact * 10000"
    )


# ---------------------------------------------------------------------------
# optimalExecution
# ---------------------------------------------------------------------------

def test_optimalExecution_schedule_sums_to_total():
    """The execution schedule must sum to totalShares."""
    result = sq.liquidity.optimalExecution(
        totalShares=10000, T=10, adv=50000, sigma=0.02
    )
    assert isinstance(result, dict), "optimalExecution must return a dict"
    assert abs(np.sum(result['schedule']) - 10000.0) < 1e-4, (
        f"Schedule sums to {np.sum(result['schedule'])}, expected 10000"
    )


def test_optimalExecution_kappa_positive():
    """kappa must be positive."""
    result = sq.liquidity.optimalExecution(
        totalShares=5000, T=5, adv=20000, sigma=0.015
    )
    assert result['kappa'] > 0, f"Expected positive kappa, got {result['kappa']}"


def test_optimalExecution_keys():
    """All required keys must be present."""
    result = sq.liquidity.optimalExecution(
        totalShares=1000, T=5, adv=10000, sigma=0.02
    )
    for key in ('schedule', 'trajectory', 'expectedCost',
                'expectedVariance', 'kappa'):
        assert key in result, f"Missing key '{key}' in optimalExecution result"


def test_optimalExecution_trajectory_starts_at_total():
    """Trajectory must start at totalShares."""
    total = 8000.0
    result = sq.liquidity.optimalExecution(
        totalShares=total, T=8, adv=40000, sigma=0.02
    )
    assert abs(result['trajectory'][0] - total) < 1e-6, (
        f"Trajectory must start at {total}, got {result['trajectory'][0]}"
    )


def test_optimalExecution_schedule_non_negative():
    """All scheduled trades must be non-negative (liquidation)."""
    result = sq.liquidity.optimalExecution(
        totalShares=3000, T=6, adv=15000, sigma=0.02
    )
    assert np.all(result['schedule'] >= -1e-9), (
        "All schedule entries must be non-negative"
    )


# ---------------------------------------------------------------------------
# thinMarketScore
# ---------------------------------------------------------------------------

def test_thinMarketScore_in_range():
    """Score must be in [0, 1]."""
    trades = [make_trade(f'2026-01-{i:02d}', 250.0 + i, 10.0 + i)
              for i in range(1, 16)]
    result = sq.liquidity.thinMarketScore(trades, window=30)
    assert isinstance(result, dict), "thinMarketScore must return a dict"
    assert 0.0 <= result['score'] <= 1.0, (
        f"score={result['score']} outside [0, 1]"
    )


def test_thinMarketScore_empty():
    """Empty trade list must return score = 0."""
    result = sq.liquidity.thinMarketScore([], window=30)
    assert result['score'] == 0.0, (
        f"Expected score=0.0 for empty trades, got {result['score']}"
    )


def test_thinMarketScore_keys():
    """All required keys must be present."""
    trades = [make_trade('2026-01-01', 200.0, 5.0)]
    result = sq.liquidity.thinMarketScore(trades, window=30)
    for key in ('score', 'nTrades', 'avgVolume', 'priceCV', 'window'):
        assert key in result, f"Missing key '{key}' in thinMarketScore result"


def test_thinMarketScore_stable_prices_higher_score():
    """Stable prices (low CV) should score higher than volatile prices."""
    stable_trades = [make_trade(f'2026-01-{i:02d}', 250.0, 10.0)
                     for i in range(1, 11)]
    volatile_trades = [make_trade(f'2026-01-{i:02d}', 250.0 + i * 20, 10.0)
                       for i in range(1, 11)]
    r_stable = sq.liquidity.thinMarketScore(stable_trades, window=30)
    r_volatile = sq.liquidity.thinMarketScore(volatile_trades, window=30)
    assert r_stable['score'] >= r_volatile['score'], (
        "Stable prices should produce a higher liquidity score"
    )


# ---------------------------------------------------------------------------
# concentrationRisk
# ---------------------------------------------------------------------------

def test_concentrationRisk_hhi_range():
    """HHI must be in (0, 1] for non-zero positions."""
    positions = np.array([100.0, 200.0, 50.0, 300.0])
    volumes = np.array([1000.0, 2000.0, 500.0, 3000.0])
    result = sq.liquidity.concentrationRisk(positions, volumes)
    assert isinstance(result, dict), "concentrationRisk must return a dict"
    assert 0.0 < result['hhi'] <= 1.0, (
        f"hhi={result['hhi']} outside (0, 1]"
    )


def test_concentrationRisk_single_position():
    """A single position must yield HHI = 1.0."""
    result = sq.liquidity.concentrationRisk(
        np.array([500.0]), np.array([1000.0])
    )
    assert abs(result['hhi'] - 1.0) < 1e-9, (
        f"Single position should have HHI=1.0, got {result['hhi']}"
    )


def test_concentrationRisk_keys():
    """All required keys must be present."""
    positions = np.array([100.0, 200.0])
    volumes = np.array([1000.0, 2000.0])
    result = sq.liquidity.concentrationRisk(positions, volumes)
    for key in ('hhi', 'participationRates', 'concentrationScore'):
        assert key in result, f"Missing key '{key}' in concentrationRisk result"


def test_concentrationRisk_equal_positions():
    """Equal positions must produce HHI = 1/n."""
    n = 4
    positions = np.ones(n) * 100.0
    volumes = np.ones(n) * 1000.0
    result = sq.liquidity.concentrationRisk(positions, volumes)
    expected_hhi = 1.0 / n
    assert abs(result['hhi'] - expected_hhi) < 1e-9, (
        f"Equal positions: expected HHI={expected_hhi}, got {result['hhi']}"
    )


def test_concentrationRisk_participation_rates():
    """participationRates must equal |pos_i| / vol_i."""
    positions = np.array([100.0, 200.0])
    volumes = np.array([500.0, 1000.0])
    result = sq.liquidity.concentrationRisk(positions, volumes)
    expected = positions / volumes
    np.testing.assert_allclose(result['participationRates'], expected, rtol=1e-9)


# ---------------------------------------------------------------------------
# optimalLiquidation
# ---------------------------------------------------------------------------

def test_optimalLiquidation_estimated_slippage_positive():
    """estimatedSlippage must be positive for a non-zero position."""
    result = sq.liquidity.optimalLiquidation(
        position=5000, adv=20000, sigma=0.02, timeHorizon=10
    )
    assert isinstance(result, dict), "optimalLiquidation must return a dict"
    assert result['estimatedSlippage'] > 0, (
        f"Expected positive estimatedSlippage, got {result['estimatedSlippage']}"
    )


def test_optimalLiquidation_keys():
    """All required keys must be present."""
    result = sq.liquidity.optimalLiquidation(
        position=1000, adv=5000, sigma=0.02, timeHorizon=5
    )
    for key in ('twapCost', 'vwapCost', 'liquidationSchedule',
                'estimatedSlippage', 'marketImpactCost'):
        assert key in result, f"Missing key '{key}' in optimalLiquidation result"


def test_optimalLiquidation_schedule_uniform():
    """Liquidation schedule must be uniform (all entries equal)."""
    n = 10
    result = sq.liquidity.optimalLiquidation(
        position=1000, adv=5000, sigma=0.02, timeHorizon=n
    )
    schedule = result['liquidationSchedule']
    assert len(schedule) == n, f"Expected schedule length={n}, got {len(schedule)}"
    assert np.all(schedule == schedule[0]), "Liquidation schedule must be uniform"


def test_optimalLiquidation_vwap_leq_twap():
    """VWAP cost must be less than or equal to TWAP cost."""
    result = sq.liquidity.optimalLiquidation(
        position=2000, adv=10000, sigma=0.02, timeHorizon=5
    )
    assert result['vwapCost'] <= result['twapCost'], (
        f"vwapCost {result['vwapCost']} should be <= twapCost {result['twapCost']}"
    )


def test_optimalLiquidation_larger_position_higher_cost():
    """Doubling the position size must increase estimated slippage."""
    r1 = sq.liquidity.optimalLiquidation(1000, 5000, 0.02, 5)
    r2 = sq.liquidity.optimalLiquidation(2000, 5000, 0.02, 5)
    assert r2['estimatedSlippage'] > r1['estimatedSlippage'], (
        "Larger position must produce higher slippage"
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
