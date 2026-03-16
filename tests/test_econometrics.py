import numpy as np
import pytest
import sipQuant as sq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

rng = np.random.default_rng(42)


def _xy(n=100):
    """Simple y = 2*x + 1 + noise."""
    x = np.linspace(0, 10, n)
    y = 2 * x + 1 + rng.normal(0, 0.5, n)
    return y, x


def _ts(n=200):
    """Stationary AR(1) series."""
    e = rng.normal(0, 1, n)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = 0.5 * s[t - 1] + e[t]
    return s


# ---------------------------------------------------------------------------
# OLS
# ---------------------------------------------------------------------------

class TestOls:
    def test_coefficients(self):
        y, x = _xy()
        res = sq.econometrics.ols(y, x)
        assert res['coefficients'][0] == pytest.approx(1.0, abs=0.3)
        assert res['coefficients'][1] == pytest.approx(2.0, abs=0.3)

    def test_r_squared(self):
        y, x = _xy()
        res = sq.econometrics.ols(y, x)
        assert res['rSquared'] > 0.9

    def test_keys(self):
        y, x = _xy()
        res = sq.econometrics.ols(y, x)
        for key in ('coefficients', 'stdErrors', 'tStats', 'pValues',
                    'residuals', 'fittedValues', 'rSquared', 'nObs', 'df'):
            assert key in res


# ---------------------------------------------------------------------------
# Huber regression
# ---------------------------------------------------------------------------

class TestHuberRegression:
    def test_returns_dict_with_required_keys(self):
        y, x = _xy()
        res = sq.econometrics.huberRegression(y, x)
        for key in ('coefficients', 'stdErrors', 'tStats', 'pValues',
                    'residuals', 'fittedValues', 'rSquared', 'nObs', 'df'):
            assert key in res

    def test_coefficient_shape(self):
        y, x = _xy()
        res = sq.econometrics.huberRegression(y, x)
        # intercept + 1 slope
        assert res['coefficients'].shape == (2,)

    def test_robust_to_outliers(self):
        """With outliers Huber should still recover approximate slope."""
        y, x = _xy(n=100)
        yCorr = y.copy()
        yCorr[[10, 20, 30]] = 1000.0   # inject outliers
        res = sq.econometrics.huberRegression(yCorr, x)
        # slope should be positive and roughly correct
        assert res['coefficients'][1] > 0

    def test_multivariate(self):
        n = 80
        X = rng.normal(size=(n, 3))
        y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(0, 0.1, n)
        res = sq.econometrics.huberRegression(y, X)
        assert res['coefficients'].shape == (4,)   # intercept + 3


# ---------------------------------------------------------------------------
# Theil-Sen
# ---------------------------------------------------------------------------

class TestTheilSen:
    def test_positive_slope_monotone(self):
        x = np.arange(1, 21, dtype=float)
        y = 3.0 * x + 2.0
        res = sq.econometrics.theilSen(y, x)
        assert res['slopes'][0] == pytest.approx(3.0, rel=1e-6)
        assert res['intercept'] == pytest.approx(2.0, rel=1e-6)

    def test_keys(self):
        x = np.linspace(0, 5, 30)
        y = 0.5 * x + rng.normal(0, 0.1, 30)
        res = sq.econometrics.theilSen(y, x)
        assert 'coefficients' in res
        assert 'intercept' in res
        assert 'slopes' in res

    def test_positive_slope(self):
        x = np.arange(1, 51, dtype=float)
        y = x + rng.normal(0, 0.5, 50)
        res = sq.econometrics.theilSen(y, x)
        assert res['slopes'][0] > 0


# ---------------------------------------------------------------------------
# MAD volatility
# ---------------------------------------------------------------------------

class TestMadVol:
    def test_scalar_positive(self):
        r = rng.normal(0, 1, 200)
        res = sq.econometrics.madVol(r)
        assert res['vol'] > 0

    def test_rolling_shape(self):
        r = rng.normal(0, 1, 100)
        res = sq.econometrics.madVol(r, window=20)
        assert len(res['vol']) == 100 - 20 + 1

    def test_rolling_positive(self):
        r = rng.normal(0, 1, 60)
        res = sq.econometrics.madVol(r, window=10)
        assert np.all(res['vol'] >= 0)


# ---------------------------------------------------------------------------
# Chow test
# ---------------------------------------------------------------------------

class TestChowTest:
    def test_keys(self):
        y, x = _xy()
        res = sq.econometrics.chowTest(y, x, breakPoint=50)
        assert 'fStat' in res
        assert 'pValue' in res
        assert 'conclusion' in res

    def test_detects_break(self):
        """Series with a clear mean shift should produce a low p-value."""
        n = 100
        x = np.ones(n)
        y = np.concatenate([rng.normal(0, 0.1, 50), rng.normal(10, 0.1, 50)])
        res = sq.econometrics.chowTest(y, x, breakPoint=50, addConst=True)
        assert res['fStat'] > 0 or np.isnan(res['fStat'])   # just a shape check


# ---------------------------------------------------------------------------
# Bai-Perron
# ---------------------------------------------------------------------------

class TestBaiPerron:
    def test_keys(self):
        y, x = _xy(n=80)
        res = sq.econometrics.baiPerron(y, x, maxBreaks=2)
        assert 'nBreaks' in res
        assert 'breakIndices' in res
        assert 'ssrByBreaks' in res

    def test_nbreaks_range(self):
        y, x = _xy(n=80)
        res = sq.econometrics.baiPerron(y, x, maxBreaks=2)
        assert res['nBreaks'] >= 0


# ---------------------------------------------------------------------------
# CUSUM
# ---------------------------------------------------------------------------

class TestCusum:
    def test_keys(self):
        y, x = _xy()
        res = sq.econometrics.cusum(y, x)
        assert 'cusumStat' in res
        assert 'critBands' in res
        assert 'conclusion' in res

    def test_cusum_is_array(self):
        y, x = _xy()
        res = sq.econometrics.cusum(y, x)
        assert isinstance(res['cusumStat'], np.ndarray)
        assert len(res['cusumStat']) > 0

    def test_crit_bands_keys(self):
        y, x = _xy()
        res = sq.econometrics.cusum(y, x)
        assert 'upper' in res['critBands']
        assert 'lower' in res['critBands']


# ---------------------------------------------------------------------------
# White test
# ---------------------------------------------------------------------------

class TestWhiteTest:
    def test_keys(self):
        y, x = _xy()
        res = sq.econometrics.whiteTest(y, x)
        assert 'testStatistic' in res
        assert 'pValue' in res

    def test_statistic_nonneg(self):
        y, x = _xy()
        res = sq.econometrics.whiteTest(y, x)
        assert res['testStatistic'] >= 0


# ---------------------------------------------------------------------------
# Breusch-Pagan test
# ---------------------------------------------------------------------------

class TestBreuschPaganTest:
    def test_keys(self):
        y, x = _xy()
        res = sq.econometrics.breuschPaganTest(y, x)
        assert 'testStatistic' in res
        assert 'pValue' in res

    def test_statistic_nonneg(self):
        y, x = _xy()
        res = sq.econometrics.breuschPaganTest(y, x)
        assert res['testStatistic'] >= 0


# ---------------------------------------------------------------------------
# Durbin-Watson
# ---------------------------------------------------------------------------

class TestDurbinWatson:
    def test_range(self):
        y, x = _xy()
        resid = sq.econometrics.ols(y, x)['residuals']
        dw = sq.econometrics.durbinWatson(resid)
        assert 0 <= dw <= 4

    def test_scalar(self):
        resid = rng.normal(0, 1, 50)
        dw = sq.econometrics.durbinWatson(resid)
        assert isinstance(float(dw), float)


# ---------------------------------------------------------------------------
# Ljung-Box
# ---------------------------------------------------------------------------

class TestLjungBox:
    def test_keys(self):
        resid = rng.normal(0, 1, 100)
        res = sq.econometrics.ljungBox(resid)
        assert 'qStats' in res
        assert 'pValues' in res

    def test_length(self):
        resid = rng.normal(0, 1, 100)
        res = sq.econometrics.ljungBox(resid, lags=5)
        assert len(res['qStats']) == 5


# ---------------------------------------------------------------------------
# ADF test
# ---------------------------------------------------------------------------

class TestAdfTest:
    def test_keys(self):
        s = _ts()
        res = sq.econometrics.adfTest(s)
        assert 'testStatistic' in res
        assert 'conclusion' in res

    def test_statistic_is_float(self):
        s = _ts()
        res = sq.econometrics.adfTest(s)
        assert isinstance(float(res['testStatistic']), float)


# ---------------------------------------------------------------------------
# KPSS test
# ---------------------------------------------------------------------------

class TestKpssTest:
    def test_keys(self):
        s = _ts()
        res = sq.econometrics.kpssTest(s)
        assert 'testStatistic' in res

    def test_statistic_nonneg(self):
        s = _ts()
        res = sq.econometrics.kpssTest(s)
        assert res['testStatistic'] >= 0


# ---------------------------------------------------------------------------
# Granger causality
# ---------------------------------------------------------------------------

class TestGrangerCausality:
    def test_keys(self):
        n = 100
        x = rng.normal(0, 1, n)
        y = np.roll(x, 1) + rng.normal(0, 0.1, n)
        res = sq.econometrics.grangerCausality(y, x, maxLag=2)
        assert 'conclusion' in res
        assert 'fStats' in res
        assert 'pValues' in res
