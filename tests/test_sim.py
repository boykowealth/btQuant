import numpy as np
import pytest
import sipQuant as sq


# ---------------------------------------------------------------------------
# Fixtures / constants
# ---------------------------------------------------------------------------

STEPS = 50
SIMS = 20
RNG = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# GBM
# ---------------------------------------------------------------------------

class TestGbm:
    def test_shape(self):
        out = sq.sim.gbm(mu=0.05, sigma=0.2, nSteps=STEPS, nSims=SIMS)
        assert out.shape == (SIMS, STEPS)

    def test_all_positive(self):
        out = sq.sim.gbm(mu=0.05, sigma=0.2, nSteps=STEPS, nSims=SIMS, s0=100.0)
        assert np.all(out > 0)

    def test_starts_near_s0(self):
        s0 = 50.0
        out = sq.sim.gbm(mu=0.0, sigma=0.01, nSteps=STEPS, nSims=SIMS, s0=s0)
        # First column should be very close to s0 for low sigma
        assert np.mean(out[:, 0]) == pytest.approx(s0, rel=0.1)


# ---------------------------------------------------------------------------
# OU
# ---------------------------------------------------------------------------

class TestOu:
    def test_shape(self):
        out = sq.sim.ou(theta=1.0, mu=0.0, sigma=0.2, nSteps=STEPS, nSims=SIMS)
        assert out.shape == (SIMS, STEPS)

    def test_float_values(self):
        out = sq.sim.ou(theta=1.0, mu=0.0, sigma=0.2, nSteps=STEPS, nSims=SIMS)
        assert out.dtype.kind == 'f'

    def test_initial_value(self):
        x0 = 3.14
        out = sq.sim.ou(theta=1.0, mu=0.0, sigma=0.0, nSteps=STEPS, nSims=SIMS, x0=x0)
        # With sigma=0 the process decays deterministically from x0
        assert out[:, 0] == pytest.approx(x0)


# ---------------------------------------------------------------------------
# Lévy-OU
# ---------------------------------------------------------------------------

class TestLevyOu:
    def test_shape(self):
        out = sq.sim.levyOu(
            theta=1.0, mu=0.0, sigma=0.2,
            jumpLambda=5.0, jumpMu=0.0, jumpSigma=0.1,
            nSteps=STEPS, nSims=SIMS,
        )
        assert out.shape == (SIMS, STEPS)

    def test_finite_values(self):
        out = sq.sim.levyOu(
            theta=1.0, mu=0.0, sigma=0.2,
            jumpLambda=5.0, jumpMu=0.0, jumpSigma=0.1,
            nSteps=STEPS, nSims=SIMS,
        )
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# ARMA
# ---------------------------------------------------------------------------

class TestArma:
    def test_shape(self):
        out = sq.sim.arma(
            arCoefs=[0.5], maCoefs=[0.2], sigma=1.0,
            nSteps=STEPS, nSims=SIMS,
        )
        assert out.shape == (SIMS, STEPS)

    def test_shape_ar2_ma1(self):
        out = sq.sim.arma(
            arCoefs=[0.3, 0.2], maCoefs=[0.1], sigma=0.5,
            nSteps=STEPS, nSims=SIMS,
        )
        assert out.shape == (SIMS, STEPS)

    def test_float_values(self):
        out = sq.sim.arma(
            arCoefs=[0.4], maCoefs=[], sigma=1.0,
            nSteps=STEPS, nSims=SIMS,
        )
        assert out.dtype.kind == 'f'


# ---------------------------------------------------------------------------
# Markov switching
# ---------------------------------------------------------------------------

class TestMarkovSwitching:
    def test_shape(self):
        out = sq.sim.markovSwitching(
            mu1=0.0, sigma1=1.0, mu2=0.5, sigma2=2.0,
            p11=0.95, p22=0.90,
            nSteps=STEPS, nSims=SIMS,
        )
        assert out.shape == (SIMS, STEPS)

    def test_finite_values(self):
        out = sq.sim.markovSwitching(
            mu1=0.0, sigma1=1.0, mu2=0.5, sigma2=2.0,
            p11=0.95, p22=0.90,
            nSteps=STEPS, nSims=SIMS,
        )
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# GARCH
# ---------------------------------------------------------------------------

class TestGarch:
    def test_shape(self):
        out = sq.sim.garch(omega=0.0001, alpha1=0.1, beta1=0.85, nSteps=STEPS, nSims=SIMS)
        assert out.shape == (SIMS, STEPS)

    def test_finite_values(self):
        out = sq.sim.garch(omega=0.0001, alpha1=0.1, beta1=0.85, nSteps=STEPS, nSims=SIMS)
        assert np.all(np.isfinite(out))


# ---------------------------------------------------------------------------
# Heston
# ---------------------------------------------------------------------------

class TestHeston:
    def test_returns_dict(self):
        res = sq.sim.heston(
            mu=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            nSteps=STEPS, nSims=SIMS,
        )
        assert isinstance(res, dict)

    def test_prices_key(self):
        res = sq.sim.heston(
            mu=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            nSteps=STEPS, nSims=SIMS,
        )
        assert 'prices' in res

    def test_variances_key(self):
        res = sq.sim.heston(
            mu=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            nSteps=STEPS, nSims=SIMS,
        )
        assert 'variances' in res

    def test_prices_shape(self):
        res = sq.sim.heston(
            mu=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            nSteps=STEPS, nSims=SIMS,
        )
        assert res['prices'].shape == (SIMS, STEPS)

    def test_variances_shape(self):
        res = sq.sim.heston(
            mu=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            nSteps=STEPS, nSims=SIMS,
        )
        assert res['variances'].shape == (SIMS, STEPS)

    def test_variances_nonneg(self):
        res = sq.sim.heston(
            mu=0.05, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7,
            nSteps=STEPS, nSims=SIMS,
        )
        assert np.all(res['variances'] >= 0)


# ---------------------------------------------------------------------------
# Compound Poisson
# ---------------------------------------------------------------------------

class TestCompoundPoisson:
    def test_shape(self):
        out = sq.sim.compoundPoisson(
            lambdaRate=5.0, jumpMu=0.0, jumpSigma=1.0,
            nSteps=STEPS, nSims=SIMS,
        )
        assert out.shape == (SIMS, STEPS)

    def test_finite_values(self):
        out = sq.sim.compoundPoisson(
            lambdaRate=5.0, jumpMu=0.0, jumpSigma=1.0,
            nSteps=STEPS, nSims=SIMS,
        )
        assert np.all(np.isfinite(out))

    def test_initial_value(self):
        s0 = 200.0
        out = sq.sim.compoundPoisson(
            lambdaRate=0.0, jumpMu=0.0, jumpSigma=1.0,
            nSteps=STEPS, nSims=SIMS, s0=s0,
        )
        assert np.all(out[:, 0] == s0)


# ---------------------------------------------------------------------------
# simulate dispatcher
# ---------------------------------------------------------------------------

class TestSimulate:
    def test_gbm_via_dispatcher(self):
        params = {'mu': 0.05, 'sigma': 0.2, 's0': 100.0}
        out = sq.sim.simulate('gbm', params, nSteps=STEPS, nSims=SIMS)
        assert out.shape == (SIMS, STEPS)

    def test_gbm_alias_case_insensitive(self):
        params = {'mu': 0.05, 'sigma': 0.2}
        out = sq.sim.simulate('GBM', params, nSteps=STEPS, nSims=SIMS)
        assert out.shape == (SIMS, STEPS)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            sq.sim.simulate('unknown', {}, nSteps=STEPS, nSims=SIMS)

    def test_heston_via_dispatcher(self):
        params = {'mu': 0.05, 'kappa': 2.0, 'theta': 0.04, 'sigma': 0.3, 'rho': -0.7}
        res = sq.sim.simulate('heston', params, nSteps=STEPS, nSims=SIMS)
        assert isinstance(res, dict)
        assert 'prices' in res
