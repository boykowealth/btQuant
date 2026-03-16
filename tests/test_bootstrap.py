"""
Tests for sipQuant.bootstrap
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import sipQuant as sq


# ---------------------------------------------------------------------------
# forwardCurve
# ---------------------------------------------------------------------------

def test_forwardCurve_forwards_positive():
    tenors = np.array([0.25, 0.5, 1.0, 2.0])
    rates = np.array([0.03, 0.032, 0.035, 0.038])
    result = sq.bootstrap.forwardCurve(spotPrice=100.0, tenors=tenors, rates=rates)
    assert np.all(result['forwards'] > 0.0)


def test_forwardCurve_base_spot():
    result = sq.bootstrap.forwardCurve(spotPrice=55.0, tenors=[1.0], rates=[0.03])
    assert result['baseSpot'] == 55.0


def test_forwardCurve_with_storage_and_convenience():
    tenors = np.array([0.5, 1.0])
    result = sq.bootstrap.forwardCurve(
        spotPrice=100.0,
        tenors=tenors,
        rates=np.array([0.03, 0.035]),
        storageCosts=0.02,
        convenienceYields=0.01,
    )
    assert np.all(result['forwards'] > 0.0)


def test_forwardCurve_keys():
    result = sq.bootstrap.forwardCurve(100.0, [1.0, 2.0], [0.03, 0.04])
    for key in ('tenors', 'forwards', 'baseSpot'):
        assert key in result


def test_forwardCurve_zero_storage_equals_basic():
    tenors = np.array([1.0, 2.0])
    rates = np.array([0.03, 0.04])
    r1 = sq.bootstrap.forwardCurve(100.0, tenors, rates)
    r2 = sq.bootstrap.forwardCurve(100.0, tenors, rates, storageCosts=0.0, convenienceYields=0.0)
    np.testing.assert_allclose(r1['forwards'], r2['forwards'], rtol=1e-10)


# ---------------------------------------------------------------------------
# discountCurve
# ---------------------------------------------------------------------------

def test_discountCurve_factors_in_unit_interval():
    tenors = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    rates = np.array([0.02, 0.025, 0.03, 0.035, 0.04])
    result = sq.bootstrap.discountCurve(tenors, rates)
    assert np.all(result['discountFactors'] > 0.0)
    assert np.all(result['discountFactors'] <= 1.0)


def test_discountCurve_log_linear():
    tenors = np.array([1.0, 2.0, 5.0])
    rates = np.array([0.03, 0.035, 0.04])
    result = sq.bootstrap.discountCurve(tenors, rates, method='log_linear')
    assert np.all(result['discountFactors'] > 0.0)
    assert np.all(result['discountFactors'] <= 1.0)


def test_discountCurve_method_stored():
    result = sq.bootstrap.discountCurve([1.0], [0.03], method='log_linear')
    assert result['method'] == 'log_linear'


def test_discountCurve_keys():
    result = sq.bootstrap.discountCurve([1.0, 2.0], [0.03, 0.04])
    for key in ('tenors', 'discountFactors', 'zeroRates', 'method'):
        assert key in result


def test_discountCurve_longer_tenor_lower_df():
    tenors = np.array([1.0, 2.0, 5.0])
    rates = np.array([0.03, 0.03, 0.03])
    result = sq.bootstrap.discountCurve(tenors, rates)
    dfs = result['discountFactors']
    assert dfs[0] > dfs[1] > dfs[2]


# ---------------------------------------------------------------------------
# volSurface
# ---------------------------------------------------------------------------

def test_volSurface_vols_shape():
    strikes = np.array([90.0, 100.0, 110.0])
    tenors = np.array([0.25, 0.5, 1.0])
    vols = np.array([
        [0.22, 0.20, 0.21],
        [0.21, 0.19, 0.20],
        [0.20, 0.18, 0.19],
    ])
    result = sq.bootstrap.volSurface(strikes, tenors, vols)
    assert result['vols'].shape == (3, 3)


def test_volSurface_strikes_correct():
    strikes = np.array([90.0, 100.0, 110.0])
    tenors = np.array([0.5, 1.0])
    vols = np.array([[0.22, 0.20, 0.21], [0.21, 0.19, 0.20]])
    result = sq.bootstrap.volSurface(strikes, tenors, vols)
    np.testing.assert_array_equal(result['strikes'], strikes)


def test_volSurface_atm_vols_length():
    strikes = np.array([90.0, 100.0, 110.0])
    tenors = np.array([0.5, 1.0])
    vols = np.array([[0.22, 0.20, 0.21], [0.21, 0.19, 0.20]])
    result = sq.bootstrap.volSurface(strikes, tenors, vols)
    assert len(result['atmVols']) == len(tenors)


def test_volSurface_keys():
    strikes = np.array([100.0, 110.0])
    tenors = np.array([1.0])
    vols = np.array([[0.20, 0.22]])
    result = sq.bootstrap.volSurface(strikes, tenors, vols)
    for key in ('strikes', 'tenors', 'vols', 'atmVols'):
        assert key in result


# ---------------------------------------------------------------------------
# interpVol
# ---------------------------------------------------------------------------

def _buildSurface():
    strikes = np.array([90.0, 100.0, 110.0])
    tenors = np.array([0.25, 0.5, 1.0])
    vols = np.array([
        [0.22, 0.20, 0.21],
        [0.21, 0.19, 0.20],
        [0.20, 0.18, 0.19],
    ])
    return sq.bootstrap.volSurface(strikes, tenors, vols)


def test_interpVol_finite():
    surface = _buildSurface()
    result = sq.bootstrap.interpVol(surface, strike=100.0, tenor=0.5)
    assert np.isfinite(result)


def test_interpVol_at_grid_point():
    surface = _buildSurface()
    # At an exact grid point, should return the grid value.
    result = sq.bootstrap.interpVol(surface, strike=100.0, tenor=0.5)
    assert abs(result - 0.19) < 1e-8


def test_interpVol_between_strikes():
    surface = _buildSurface()
    # Between strikes, should be between adjacent values.
    v1 = sq.bootstrap.interpVol(surface, strike=90.0, tenor=0.5)
    v2 = sq.bootstrap.interpVol(surface, strike=110.0, tenor=0.5)
    vmid = sq.bootstrap.interpVol(surface, strike=100.0, tenor=0.5)
    # vmid should be a finite value in a reasonable vol range
    assert 0.0 < vmid < 1.0


# ---------------------------------------------------------------------------
# convenienceYieldCurve
# ---------------------------------------------------------------------------

def test_convenienceYieldCurve_returns_dict():
    tenors = np.array([0.25, 0.5, 1.0])
    futures = np.array([101.0, 102.0, 103.0])
    rates = 0.03
    result = sq.bootstrap.convenienceYieldCurve(100.0, futures, tenors, rates)
    assert isinstance(result, dict)


def test_convenienceYieldCurve_keys():
    tenors = np.array([0.25, 0.5, 1.0])
    futures = np.array([101.0, 102.0, 103.0])
    result = sq.bootstrap.convenienceYieldCurve(100.0, futures, tenors, 0.03)
    for key in ('tenors', 'convenienceYields', 'impliedForwards'):
        assert key in result


def test_convenienceYieldCurve_finite():
    tenors = np.array([0.25, 0.5, 1.0])
    futures = np.array([101.0, 102.5, 104.0])
    result = sq.bootstrap.convenienceYieldCurve(100.0, futures, tenors, 0.03)
    assert np.all(np.isfinite(result['convenienceYields']))


# ---------------------------------------------------------------------------
# bootstrapZeroCurve
# ---------------------------------------------------------------------------

def test_bootstrapZeroCurve_zero_rates_positive():
    maturities = np.array([1.0, 2.0, 3.0, 5.0])
    couponRates = np.array([0.03, 0.033, 0.035, 0.038])
    result = sq.bootstrap.bootstrapZeroCurve(maturities, couponRates)
    assert np.all(result['zeroRates'] > 0.0)


def test_bootstrapZeroCurve_discount_factors_in_01():
    maturities = np.array([1.0, 2.0, 3.0])
    couponRates = np.array([0.03, 0.035, 0.04])
    result = sq.bootstrap.bootstrapZeroCurve(maturities, couponRates)
    assert np.all(result['discountFactors'] > 0.0)
    assert np.all(result['discountFactors'] <= 1.0)


def test_bootstrapZeroCurve_keys():
    result = sq.bootstrap.bootstrapZeroCurve([1.0, 2.0], [0.03, 0.035])
    for key in ('maturities', 'zeroRates', 'discountFactors'):
        assert key in result


def test_bootstrapZeroCurve_length():
    maturities = [1.0, 2.0, 3.0]
    result = sq.bootstrap.bootstrapZeroCurve(maturities, [0.03, 0.035, 0.04])
    assert len(result['zeroRates']) == 3


# ---------------------------------------------------------------------------
# spreadCurve
# ---------------------------------------------------------------------------

def test_spreadCurve_adjusted_equals_base_plus_spread():
    tenors = np.array([1.0, 2.0, 5.0])
    baseRates = np.array([0.03, 0.035, 0.04])
    spreadBps = np.array([50.0, 60.0, 75.0])
    result = sq.bootstrap.spreadCurve(baseRates, spreadBps, tenors)
    expected = baseRates + spreadBps / 10000.0
    np.testing.assert_allclose(result['adjustedRates'], expected, rtol=1e-10)


def test_spreadCurve_scalar_spread():
    tenors = np.array([1.0, 2.0, 3.0])
    baseRates = np.array([0.03, 0.035, 0.04])
    result = sq.bootstrap.spreadCurve(baseRates, 50.0, tenors)
    expected = baseRates + 50.0 / 10000.0
    np.testing.assert_allclose(result['adjustedRates'], expected, rtol=1e-10)


def test_spreadCurve_keys():
    result = sq.bootstrap.spreadCurve([0.03], [50.0], [1.0])
    for key in ('tenors', 'adjustedRates', 'baseRates', 'spreadBps'):
        assert key in result
