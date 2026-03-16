"""
Tests for sipQuant.risk
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import sipQuant as sq


def _returns(T=500, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(T) * 0.01


def _prices(T=500, seed=42):
    r = _returns(T, seed)
    return np.cumprod(1.0 + r) * 100.0


def _matrixReturns(T=500, n=4, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n)) * 0.01


# ---------------------------------------------------------------------------
# var
# ---------------------------------------------------------------------------

def test_var_historical_positive():
    r = _returns()
    result = sq.risk.var(r, alpha=0.05, method='historical')
    assert result['var'] > 0.0, "VaR should be a positive loss"


def test_var_parametric_positive():
    r = _returns()
    result = sq.risk.var(r, alpha=0.05, method='parametric')
    assert result['var'] > 0.0


def test_var_cornish_fisher_positive():
    r = _returns()
    result = sq.risk.var(r, alpha=0.05, method='cornish_fisher')
    assert result['var'] > 0.0


def test_var_method_stored():
    r = _returns()
    result = sq.risk.var(r, method='historical')
    assert result['method'] == 'historical'


# ---------------------------------------------------------------------------
# cvar
# ---------------------------------------------------------------------------

def test_cvar_ge_var():
    r = _returns()
    cvarResult = sq.risk.cvar(r, alpha=0.05, method='historical')
    varResult = sq.risk.var(r, alpha=0.05, method='historical')
    assert cvarResult['cvar'] >= varResult['var'] - 1e-8, "CVaR should be >= VaR"


def test_cvar_positive():
    r = _returns()
    result = sq.risk.cvar(r)
    assert result['cvar'] > 0.0


def test_cvar_keys():
    r = _returns()
    result = sq.risk.cvar(r)
    for key in ('cvar', 'var', 'method'):
        assert key in result


# ---------------------------------------------------------------------------
# maxDrawdown
# ---------------------------------------------------------------------------

def test_maxDrawdown_negative_or_zero():
    p = _prices()
    result = sq.risk.maxDrawdown(p)
    assert result['maxDrawdown'] <= 0.0, "maxDrawdown should be <= 0"


def test_maxDrawdown_peak_before_trough():
    p = _prices()
    result = sq.risk.maxDrawdown(p)
    assert result['peakIdx'] <= result['troughIdx']


def test_maxDrawdown_drawdowns_length():
    p = _prices()
    result = sq.risk.maxDrawdown(p)
    assert len(result['drawdowns']) == len(p)


# ---------------------------------------------------------------------------
# sortino
# ---------------------------------------------------------------------------

def test_sortino_finite():
    r = _returns()
    result = sq.risk.sortino(r)
    assert np.isfinite(result['sortino'])


def test_sortino_keys():
    r = _returns()
    result = sq.risk.sortino(r)
    for key in ('sortino', 'annualReturn', 'downstdDev'):
        assert key in result


def test_sortino_downstd_positive():
    r = _returns()
    result = sq.risk.sortino(r)
    assert result['downstdDev'] >= 0.0


# ---------------------------------------------------------------------------
# calmar
# ---------------------------------------------------------------------------

def test_calmar_finite():
    r = _returns()
    result = sq.risk.calmar(r)
    assert np.isfinite(result['calmar'])


def test_calmar_keys():
    r = _returns()
    result = sq.risk.calmar(r)
    for key in ('calmar', 'annualReturn', 'maxDrawdown'):
        assert key in result


# ---------------------------------------------------------------------------
# hillEstimator
# ---------------------------------------------------------------------------

def test_hillEstimator_xi_positive():
    rng = np.random.default_rng(42)
    # Heavy-tailed returns (Pareto-like losses).
    r = -np.abs(rng.standard_cauchy(300)) * 0.005
    result = sq.risk.hillEstimator(r)
    assert result['xi'] > 0.0, "Hill estimator xi should be positive for heavy tails"


def test_hillEstimator_keys():
    r = _returns()
    result = sq.risk.hillEstimator(r)
    for key in ('xi', 'threshold', 'nExceedances'):
        assert key in result


# ---------------------------------------------------------------------------
# portfolioVar
# ---------------------------------------------------------------------------

def test_portfolioVar_has_var():
    R = _matrixReturns()
    w = np.array([0.25, 0.25, 0.25, 0.25])
    result = sq.risk.portfolioVar(w, R)
    assert 'var' in result


def test_portfolioVar_var_positive():
    R = _matrixReturns()
    w = np.array([0.25, 0.25, 0.25, 0.25])
    result = sq.risk.portfolioVar(w, R)
    assert result['var'] > 0.0


def test_portfolioVar_portfolio_returns_length():
    R = _matrixReturns()
    w = np.array([0.25, 0.25, 0.25, 0.25])
    result = sq.risk.portfolioVar(w, R)
    assert len(result['portfolioReturns']) == R.shape[0]


# ---------------------------------------------------------------------------
# rollingVol
# ---------------------------------------------------------------------------

def test_rollingVol_correct_length():
    r = _returns(T=100)
    window = 21
    result = sq.risk.rollingVol(r, window=window)
    assert len(result) == 100 - window + 1


def test_rollingVol_all_positive():
    r = _returns()
    result = sq.risk.rollingVol(r, window=21)
    assert np.all(result >= 0.0)


def test_rollingVol_finite():
    r = _returns()
    result = sq.risk.rollingVol(r)
    assert np.all(np.isfinite(result))


# ---------------------------------------------------------------------------
# beta
# ---------------------------------------------------------------------------

def test_beta_has_beta_key():
    rng = np.random.default_rng(42)
    market = rng.standard_normal(300) * 0.01
    asset = 1.2 * market + rng.standard_normal(300) * 0.005
    result = sq.risk.beta(asset, market)
    assert 'beta' in result


def test_beta_reasonable_value():
    rng = np.random.default_rng(42)
    market = rng.standard_normal(300) * 0.01
    asset = 1.2 * market + rng.standard_normal(300) * 0.002
    result = sq.risk.beta(asset, market)
    assert abs(result['beta'] - 1.2) < 0.2, f"beta {result['beta']:.3f} far from 1.2"


def test_beta_rSquared_in_unit_interval():
    rng = np.random.default_rng(42)
    market = rng.standard_normal(300) * 0.01
    asset = 1.0 * market + rng.standard_normal(300) * 0.005
    result = sq.risk.beta(asset, market)
    assert 0.0 <= result['rSquared'] <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# trackingError
# ---------------------------------------------------------------------------

def test_trackingError_has_key():
    rng = np.random.default_rng(42)
    benchmark = rng.standard_normal(300) * 0.01
    portfolio = benchmark + rng.standard_normal(300) * 0.002
    result = sq.risk.trackingError(portfolio, benchmark)
    assert 'trackingError' in result


def test_trackingError_positive():
    rng = np.random.default_rng(42)
    benchmark = rng.standard_normal(300) * 0.01
    portfolio = benchmark + rng.standard_normal(300) * 0.002
    result = sq.risk.trackingError(portfolio, benchmark)
    assert result['trackingError'] >= 0.0


def test_trackingError_keys():
    rng = np.random.default_rng(42)
    benchmark = rng.standard_normal(300) * 0.01
    portfolio = benchmark + rng.standard_normal(300) * 0.002
    result = sq.risk.trackingError(portfolio, benchmark)
    for key in ('trackingError', 'informationRatio', 'activeReturns'):
        assert key in result
