"""
Tests for sipQuant.portfolio
Seed 42 for reproducibility.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import sipQuant as sq


def _syntheticInputs(n=4, T=200, seed=42):
    rng = np.random.default_rng(seed)
    returns = rng.standard_normal((T, n)) * 0.01
    mu = returns.mean(axis=0) + 0.0005 * np.arange(1, n + 1)
    cov = np.cov(returns.T) + np.eye(n) * 1e-4
    return mu, cov, returns


# ---------------------------------------------------------------------------
# meanVariance
# ---------------------------------------------------------------------------

def test_meanVariance_longOnly_weights_sum_to_one():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.meanVariance(mu, cov, allowShort=False)
    assert abs(result['weights'].sum() - 1.0) < 1e-5, "weights should sum to 1"


def test_meanVariance_longOnly_non_negative():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.meanVariance(mu, cov, allowShort=False)
    assert np.all(result['weights'] >= -1e-6), "long-only weights should be non-negative"


def test_meanVariance_unconstrained_weights_sum_to_one():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.meanVariance(mu, cov, allowShort=True)
    assert abs(result['weights'].sum() - 1.0) < 1e-5, "weights should sum to 1"


def test_meanVariance_target_return_keys():
    mu, cov, _ = _syntheticInputs()
    targetRet = float(mu.mean())
    result = sq.portfolio.meanVariance(mu, cov, targetReturn=targetRet, allowShort=False)
    for key in ('weights', 'return', 'volatility', 'sharpe'):
        assert key in result, f"missing key '{key}'"


def test_meanVariance_volatility_positive():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.meanVariance(mu, cov)
    assert result['volatility'] > 0.0, "volatility should be positive"


# ---------------------------------------------------------------------------
# efficientFrontier
# ---------------------------------------------------------------------------

def test_efficientFrontier_returns_length():
    mu, cov, _ = _syntheticInputs()
    nPoints = 10
    result = sq.portfolio.efficientFrontier(mu, cov, nPoints=nPoints)
    assert len(result['returns']) == nPoints, "returns array should have nPoints entries"


def test_efficientFrontier_volatilities_length():
    mu, cov, _ = _syntheticInputs()
    nPoints = 10
    result = sq.portfolio.efficientFrontier(mu, cov, nPoints=nPoints)
    assert len(result['volatilities']) == nPoints


def test_efficientFrontier_weights_shape():
    mu, cov, _ = _syntheticInputs()
    n = len(mu)
    nPoints = 10
    result = sq.portfolio.efficientFrontier(mu, cov, nPoints=nPoints)
    assert result['weights'].shape == (nPoints, n)


def test_efficientFrontier_vols_positive():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.efficientFrontier(mu, cov, nPoints=5)
    assert np.all(result['volatilities'] > 0)


# ---------------------------------------------------------------------------
# hrp
# ---------------------------------------------------------------------------

def test_hrp_weights_sum_to_one():
    _, _, returns = _syntheticInputs()
    result = sq.portfolio.hrp(returns)
    assert abs(result['weights'].sum() - 1.0) < 1e-5


def test_hrp_weights_non_negative():
    _, _, returns = _syntheticInputs()
    result = sq.portfolio.hrp(returns)
    assert np.all(result['weights'] >= -1e-8)


def test_hrp_order_length():
    _, _, returns = _syntheticInputs(n=4)
    result = sq.portfolio.hrp(returns)
    assert len(result['order']) == 4


def test_hrp_order_is_permutation():
    _, _, returns = _syntheticInputs(n=4)
    result = sq.portfolio.hrp(returns)
    assert sorted(result['order']) == list(range(4))


# ---------------------------------------------------------------------------
# riskParity
# ---------------------------------------------------------------------------

def test_riskParity_weights_sum_to_one():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.riskParity(cov)
    assert abs(result['weights'].sum() - 1.0) < 1e-5


def test_riskParity_weights_positive():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.riskParity(cov)
    assert np.all(result['weights'] >= -1e-8)


def test_riskParity_risk_contributions_roughly_equal():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.riskParity(cov)
    rc = result['riskContributions']
    rc = rc / (rc.sum() + 1e-16)
    # Each RC should be within 2x of each other.
    assert rc.max() / (rc.min() + 1e-16) < 2.0, "risk contributions should be roughly equal"


def test_riskParity_keys():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.riskParity(cov)
    assert 'weights' in result and 'riskContributions' in result


# ---------------------------------------------------------------------------
# blackLitterman
# ---------------------------------------------------------------------------

def test_blackLitterman_muBL_shape():
    mu, cov, _ = _syntheticInputs(n=4)
    n = len(mu)
    P = np.array([[1, -1, 0, 0]], dtype=float)
    Q = np.array([0.001])
    result = sq.portfolio.blackLitterman(mu, cov, P, Q)
    assert result['muBL'].shape == (n,)


def test_blackLitterman_covBL_shape():
    mu, cov, _ = _syntheticInputs(n=4)
    n = len(mu)
    P = np.array([[1, -1, 0, 0]], dtype=float)
    Q = np.array([0.001])
    result = sq.portfolio.blackLitterman(mu, cov, P, Q)
    assert result['covBL'].shape == (n, n)


def test_blackLitterman_weights_present():
    mu, cov, _ = _syntheticInputs(n=4)
    P = np.eye(4)[:2]
    Q = np.array([0.001, 0.002])
    result = sq.portfolio.blackLitterman(mu, cov, P, Q)
    assert 'weights' in result


# ---------------------------------------------------------------------------
# maxSharpe
# ---------------------------------------------------------------------------

def test_maxSharpe_weights_sum_to_one():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.maxSharpe(mu, cov)
    assert abs(result['weights'].sum() - 1.0) < 1e-5


def test_maxSharpe_sharpe_finite():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.maxSharpe(mu, cov)
    assert np.isfinite(result['sharpe'])


def test_maxSharpe_volatility_positive():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.maxSharpe(mu, cov)
    assert result['volatility'] > 0.0


def test_maxSharpe_keys():
    mu, cov, _ = _syntheticInputs()
    result = sq.portfolio.maxSharpe(mu, cov)
    for key in ('weights', 'sharpe', 'return', 'volatility'):
        assert key in result


# ---------------------------------------------------------------------------
# minCvar
# ---------------------------------------------------------------------------

def test_minCvar_weights_sum_to_one():
    _, _, returns = _syntheticInputs()
    result = sq.portfolio.minCvar(returns)
    assert abs(result['weights'].sum() - 1.0) < 1e-5


def test_minCvar_weights_non_negative():
    _, _, returns = _syntheticInputs()
    result = sq.portfolio.minCvar(returns)
    assert np.all(result['weights'] >= -1e-8)


def test_minCvar_cvar_positive():
    _, _, returns = _syntheticInputs()
    result = sq.portfolio.minCvar(returns)
    assert np.isfinite(result['cvar'])


def test_minCvar_keys():
    _, _, returns = _syntheticInputs()
    result = sq.portfolio.minCvar(returns)
    for key in ('weights', 'cvar', 'var'):
        assert key in result
