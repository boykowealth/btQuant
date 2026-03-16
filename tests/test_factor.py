"""
Tests for sipQuant.factor
SIP Global (Systematic Index Partners)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import sipQuant as sq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def returnsData():
    """Synthetic market and asset returns for factor tests."""
    rng = np.random.default_rng(0)
    n = 120
    marketReturns = rng.normal(0.0008, 0.015, n)
    return n, marketReturns


# ---------------------------------------------------------------------------
# CAPM
# ---------------------------------------------------------------------------

class TestCAPM:
    def test_beta_approx_one_when_asset_equals_market(self, returnsData):
        _, mr = returnsData
        result = sq.factor.capm(mr, mr)
        assert result['beta'] == pytest.approx(1.0, abs=1e-8)

    def test_alpha_approx_zero_when_asset_equals_market(self, returnsData):
        _, mr = returnsData
        result = sq.factor.capm(mr, mr)
        assert result['alpha'] == pytest.approx(0.0, abs=1e-8)

    def test_r_squared_one_when_asset_equals_market(self, returnsData):
        _, mr = returnsData
        result = sq.factor.capm(mr, mr)
        assert result['rSquared'] == pytest.approx(1.0, abs=1e-6)

    def test_beta_scales_with_leverage(self, returnsData):
        _, mr = returnsData
        rng = np.random.default_rng(5)
        noise = rng.normal(0, 0.001, len(mr))
        asset2x = 2.0 * mr + noise
        result = sq.factor.capm(asset2x, mr)
        assert result['beta'] == pytest.approx(2.0, abs=0.05)

    def test_t_stat_significant_when_beta_equals_one(self, returnsData):
        _, mr = returnsData
        # When asset == market, t-stat for beta should be very large
        result = sq.factor.capm(mr, mr)
        assert abs(result['tStatBeta']) > 10.0

    def test_return_keys(self, returnsData):
        _, mr = returnsData
        result = sq.factor.capm(mr, mr)
        for key in ['alpha', 'beta', 'rSquared', 'tStatAlpha', 'tStatBeta',
                    'pValueAlpha', 'pValueBeta']:
            assert key in result

    def test_p_values_in_unit_interval(self, returnsData):
        _, mr = returnsData
        rng = np.random.default_rng(3)
        asset = 0.5 * mr + rng.normal(0, 0.005, len(mr))
        result = sq.factor.capm(asset, mr)
        assert 0.0 <= result['pValueAlpha'] <= 1.0
        assert 0.0 <= result['pValueBeta'] <= 1.0

    def test_rf_adjusted(self, returnsData):
        _, mr = returnsData
        rf = 0.0001
        # With same excess returns, beta still ~1
        asset = mr.copy()
        result = sq.factor.capm(asset, mr, rf=rf)
        assert result['beta'] == pytest.approx(1.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Rolling beta
# ---------------------------------------------------------------------------

class TestRollingBeta:
    def test_length_n_minus_window_plus_one(self, returnsData):
        n, mr = returnsData
        rng = np.random.default_rng(1)
        asset = 0.8 * mr + rng.normal(0, 0.005, n)
        window = 60
        result = sq.factor.rollingBeta(asset, mr, window=window)
        expected_len = n - window + 1
        assert len(result['betas']) == expected_len

    def test_alphas_length_matches_betas(self, returnsData):
        n, mr = returnsData
        rng = np.random.default_rng(2)
        asset = 1.2 * mr + rng.normal(0, 0.005, n)
        result = sq.factor.rollingBeta(asset, mr, window=40)
        assert len(result['alphas']) == len(result['betas'])

    def test_r_squared_length_matches_betas(self, returnsData):
        n, mr = returnsData
        rng = np.random.default_rng(3)
        asset = mr + rng.normal(0, 0.003, n)
        result = sq.factor.rollingBeta(asset, mr, window=30)
        assert len(result['rSquared']) == len(result['betas'])

    def test_betas_finite(self, returnsData):
        n, mr = returnsData
        rng = np.random.default_rng(4)
        asset = mr + rng.normal(0, 0.005, n)
        result = sq.factor.rollingBeta(asset, mr, window=30)
        assert np.all(np.isfinite(result['betas']))

    def test_constant_beta_asset(self, returnsData):
        n, mr = returnsData
        # Perfect beta=1.5 asset with no noise
        asset = 1.5 * mr
        result = sq.factor.rollingBeta(asset, mr, window=20)
        np.testing.assert_allclose(result['betas'], 1.5, atol=1e-8)

    def test_window_equal_to_n_gives_single_beta(self, returnsData):
        n, mr = returnsData
        asset = 0.9 * mr
        result = sq.factor.rollingBeta(asset, mr, window=n)
        assert len(result['betas']) == 1


# ---------------------------------------------------------------------------
# PCA factors
# ---------------------------------------------------------------------------

class TestPCAFactors:
    def setup_method(self):
        rng = np.random.default_rng(10)
        self.T = 100
        self.n = 10
        # Three-factor structure
        F = rng.standard_normal((self.T, 3))
        L = rng.standard_normal((self.n, 3))
        self.returns = F @ L.T + 0.05 * rng.standard_normal((self.T, self.n))

    def test_factors_shape(self):
        result = sq.factor.pcaFactors(self.returns, nFactors=3)
        assert result['factors'].shape == (self.T, 3)

    def test_loadings_shape(self):
        result = sq.factor.pcaFactors(self.returns, nFactors=3)
        assert result['loadings'].shape == (self.n, 3)

    def test_residuals_shape(self):
        result = sq.factor.pcaFactors(self.returns, nFactors=3)
        assert result['residuals'].shape == (self.T, self.n)

    def test_r_squared_by_asset_shape(self):
        result = sq.factor.pcaFactors(self.returns, nFactors=3)
        assert result['rSquaredByAsset'].shape == (self.n,)

    def test_r_squared_in_unit_interval(self):
        result = sq.factor.pcaFactors(self.returns, nFactors=3)
        assert np.all(result['rSquaredByAsset'] >= 0.0)
        assert np.all(result['rSquaredByAsset'] <= 1.0 + 1e-10)

    def test_explained_variance_decreasing(self):
        result = sq.factor.pcaFactors(self.returns, nFactors=3)
        ev = result['explainedVariance']
        assert ev[0] >= ev[1] >= ev[2]

    def test_factor_returns_alias(self):
        result = sq.factor.pcaFactors(self.returns, nFactors=3)
        # factorReturns should be identical object to factors
        np.testing.assert_array_equal(result['factorReturns'], result['factors'])

    def test_high_r_squared_for_strong_factor_structure(self):
        # With 10x amplified factor loading, R2 should be very high
        rng = np.random.default_rng(20)
        F = rng.standard_normal((self.T, 3))
        L = 10.0 * rng.standard_normal((self.n, 3))
        strongReturns = F @ L.T + 0.01 * rng.standard_normal((self.T, self.n))
        result = sq.factor.pcaFactors(strongReturns, nFactors=3)
        assert np.mean(result['rSquaredByAsset']) > 0.95
