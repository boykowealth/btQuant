"""
Tests for sipQuant.distributions
Seed 42 for reproducibility.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import sipQuant as sq


# ---------------------------------------------------------------------------
# gaussianCopula
# ---------------------------------------------------------------------------

def test_gaussianCopula_shape():
    result = sq.distributions.gaussianCopula(n=200, rho=0.5, d=2, seed=42)
    assert result.shape == (200, 2)


def test_gaussianCopula_values_in_01():
    result = sq.distributions.gaussianCopula(n=200, rho=0.5, d=2, seed=42)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_gaussianCopula_higher_dimension():
    result = sq.distributions.gaussianCopula(n=100, rho=0.3, d=4, seed=42)
    assert result.shape == (100, 4)


def test_gaussianCopula_matrix_rho():
    rho = np.array([[1.0, 0.5], [0.5, 1.0]])
    result = sq.distributions.gaussianCopula(n=100, rho=rho, d=2, seed=42)
    assert result.shape == (100, 2)


# ---------------------------------------------------------------------------
# tCopula
# ---------------------------------------------------------------------------

def test_tCopula_shape():
    result = sq.distributions.tCopula(n=200, rho=0.5, df=5.0, d=2, seed=42)
    assert result.shape == (200, 2)


def test_tCopula_values_in_01():
    result = sq.distributions.tCopula(n=200, rho=0.5, df=5.0, d=2, seed=42)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_tCopula_higher_df_resembles_gaussian():
    """Large df t-copula should be close to Gaussian copula."""
    u_t = sq.distributions.tCopula(n=500, rho=0.5, df=1000.0, d=2, seed=42)
    u_g = sq.distributions.gaussianCopula(n=500, rho=0.5, d=2, seed=42)
    # Rank correlations should be similar.
    corr_t = float(np.corrcoef(u_t[:, 0], u_t[:, 1])[0, 1])
    corr_g = float(np.corrcoef(u_g[:, 0], u_g[:, 1])[0, 1])
    assert abs(corr_t - corr_g) < 0.15


# ---------------------------------------------------------------------------
# claytonCopula
# ---------------------------------------------------------------------------

def test_claytonCopula_shape():
    result = sq.distributions.claytonCopula(n=300, theta=2.0, seed=42)
    assert result.shape == (300, 2)


def test_claytonCopula_values_in_01():
    result = sq.distributions.claytonCopula(n=300, theta=2.0, seed=42)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_claytonCopula_lower_tail_dependence():
    """Clayton copula has positive lower tail dependence."""
    result = sq.distributions.claytonCopula(n=2000, theta=3.0, seed=42)
    td = sq.distributions.tailDependence(result, threshold=0.1)
    assert td['lower'] > 0.0


# ---------------------------------------------------------------------------
# gumbelCopula
# ---------------------------------------------------------------------------

def test_gumbelCopula_shape():
    result = sq.distributions.gumbelCopula(n=300, theta=2.0, seed=42)
    assert result.shape == (300, 2)


def test_gumbelCopula_values_in_01():
    result = sq.distributions.gumbelCopula(n=300, theta=2.0, seed=42)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_gumbelCopula_theta_one_independent():
    """theta=1 gives independence copula."""
    result = sq.distributions.gumbelCopula(n=500, theta=1.0, seed=42)
    assert result.shape == (500, 2)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# tailDependence
# ---------------------------------------------------------------------------

def test_tailDependence_lower_in_01():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((500, 2))
    result = sq.distributions.tailDependence(data, threshold=0.1)
    assert 0.0 <= result['lower'] <= 1.0


def test_tailDependence_upper_in_01():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((500, 2))
    result = sq.distributions.tailDependence(data, threshold=0.1)
    assert 0.0 <= result['upper'] <= 1.0


def test_tailDependence_perfectly_correlated():
    """Perfect positive correlation implies high tail dependence."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(1000)
    data = np.column_stack([x, x])
    result = sq.distributions.tailDependence(data, threshold=0.1)
    assert result['lower'] > 0.5
    assert result['upper'] > 0.5


def test_tailDependence_keys():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((200, 2))
    result = sq.distributions.tailDependence(data)
    for key in ('lower', 'upper', 'threshold'):
        assert key in result


# ---------------------------------------------------------------------------
# kendallTau
# ---------------------------------------------------------------------------

def test_kendallTau_scalar():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    y = rng.standard_normal(50)
    result = sq.distributions.kendallTau(x, y)
    assert isinstance(result, float)


def test_kendallTau_in_neg1_1():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50)
    y = rng.standard_normal(50)
    result = sq.distributions.kendallTau(x, y)
    assert -1.0 <= result <= 1.0


def test_kendallTau_perfect_positive():
    x = np.arange(20, dtype=float)
    result = sq.distributions.kendallTau(x, x)
    assert abs(result - 1.0) < 1e-6


def test_kendallTau_perfect_negative():
    x = np.arange(20, dtype=float)
    result = sq.distributions.kendallTau(x, -x)
    assert abs(result + 1.0) < 1e-6


# ---------------------------------------------------------------------------
# spearmanRho
# ---------------------------------------------------------------------------

def test_spearmanRho_scalar():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(80)
    y = rng.standard_normal(80)
    result = sq.distributions.spearmanRho(x, y)
    assert isinstance(result, float)


def test_spearmanRho_in_neg1_1():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(80)
    y = rng.standard_normal(80)
    result = sq.distributions.spearmanRho(x, y)
    assert -1.0 <= result <= 1.0


def test_spearmanRho_perfect_positive():
    x = np.arange(30, dtype=float)
    result = sq.distributions.spearmanRho(x, x)
    assert abs(result - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# fitNormal
# ---------------------------------------------------------------------------

def test_fitNormal_keys():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(300) * 2.0 + 5.0
    result = sq.distributions.fitNormal(data)
    assert 'mu' in result and 'sigma' in result


def test_fitNormal_mu_close():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000) * 2.0 + 5.0
    result = sq.distributions.fitNormal(data)
    assert abs(result['mu'] - 5.0) < 0.3


def test_fitNormal_sigma_close():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(1000) * 2.0 + 5.0
    result = sq.distributions.fitNormal(data)
    assert abs(result['sigma'] - 2.0) < 0.3


def test_fitNormal_loglik_finite():
    rng = np.random.default_rng(42)
    data = rng.standard_normal(100)
    result = sq.distributions.fitNormal(data)
    assert np.isfinite(result['logLik'])


# ---------------------------------------------------------------------------
# fitT
# ---------------------------------------------------------------------------

def test_fitT_df_greater_than_2():
    rng = np.random.default_rng(42)
    # Generate t(5) samples.
    data = rng.standard_t(5, size=500)
    result = sq.distributions.fitT(data)
    assert result['df'] > 2.0


def test_fitT_keys():
    rng = np.random.default_rng(42)
    data = rng.standard_t(5, size=200)
    result = sq.distributions.fitT(data)
    for key in ('mu', 'sigma', 'df', 'logLik'):
        assert key in result


def test_fitT_sigma_positive():
    rng = np.random.default_rng(42)
    data = rng.standard_t(5, size=200)
    result = sq.distributions.fitT(data)
    assert result['sigma'] > 0.0


# ---------------------------------------------------------------------------
# fitGamma
# ---------------------------------------------------------------------------

def test_fitGamma_shape_positive():
    rng = np.random.default_rng(42)
    data = rng.gamma(shape=3.0, scale=2.0, size=500)
    result = sq.distributions.fitGamma(data)
    assert result['shape'] > 0.0


def test_fitGamma_scale_positive():
    rng = np.random.default_rng(42)
    data = rng.gamma(shape=3.0, scale=2.0, size=500)
    result = sq.distributions.fitGamma(data)
    assert result['scale'] > 0.0


def test_fitGamma_keys():
    rng = np.random.default_rng(42)
    data = rng.gamma(2.0, 1.5, size=300)
    result = sq.distributions.fitGamma(data)
    for key in ('shape', 'scale', 'logLik'):
        assert key in result


def test_fitGamma_shape_estimate_reasonable():
    rng = np.random.default_rng(42)
    data = rng.gamma(shape=4.0, scale=1.0, size=1000)
    result = sq.distributions.fitGamma(data)
    assert abs(result['shape'] - 4.0) < 1.0
