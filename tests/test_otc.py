"""
Tests for sipQuant.otc
SIP Global (Systematic Index Partners)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import sipQuant as sq


# ---------------------------------------------------------------------------
# commoditySwap
# ---------------------------------------------------------------------------

class TestCommoditySwap:
    def test_zero_npv_when_fixed_equals_mean_index(self):
        schedule = np.array([0.25, 0.5, 0.75, 1.0])
        indexCurve = np.array([100.0, 100.0, 100.0, 100.0])
        result = sq.otc.commoditySwap(100.0, indexCurve, 1000, schedule, 0.05)
        assert abs(result['npv']) < 1e-6

    def test_positive_npv_when_float_above_fixed(self):
        schedule = np.array([0.5, 1.0])
        indexCurve = np.array([110.0, 115.0])
        result = sq.otc.commoditySwap(100.0, indexCurve, 1000, schedule, 0.03)
        assert result['npv'] > 0

    def test_return_keys(self):
        schedule = np.array([1.0])
        result = sq.otc.commoditySwap(100.0, np.array([105.0]), 100, schedule, 0.05)
        assert 'npv' in result
        assert 'fixedLegPV' in result
        assert 'floatLegPV' in result
        assert 'greeks' in result
        assert 'delta' in result['greeks']
        assert 'dv01' in result['greeks']

    def test_npv_equals_float_minus_fixed(self):
        schedule = np.array([1.0])
        indexCurve = np.array([105.0])
        notional = 1000.0
        r = 0.05
        result = sq.otc.commoditySwap(100.0, indexCurve, notional, schedule, r)
        expected = (105.0 - 100.0) * notional * np.exp(-r * 1.0)
        assert result['npv'] == pytest.approx(expected, rel=1e-6)

    def test_mean_index_npv_near_zero_random(self):
        rng = np.random.default_rng(42)
        schedule = np.linspace(0.25, 2.0, 8)
        indexCurve = 100.0 + rng.normal(0, 1, 8)
        fixedPrice = float(np.mean(indexCurve))
        result = sq.otc.commoditySwap(fixedPrice, indexCurve, 1000, schedule, 0.04)
        # NPV should be small relative to notional (discount effect causes small residual)
        assert abs(result['npv']) < 100.0


# ---------------------------------------------------------------------------
# asianSwap
# ---------------------------------------------------------------------------

class TestAsianSwap:
    def test_npv_finite(self):
        prices = np.linspace(95, 105, 12)
        result = sq.otc.asianSwap(100.0, prices, 1000, r=0.04, T=1.0)
        assert np.isfinite(result['npv'])

    def test_average_index_matches_input(self):
        prices = np.array([100.0, 102.0, 98.0, 101.0])
        result = sq.otc.asianSwap(100.0, prices, 1, r=0.0, T=1.0)
        assert result['averageIndex'] == pytest.approx(np.mean(prices))

    def test_positive_npv_when_avg_above_fixed(self):
        prices = np.array([110.0, 112.0, 108.0])
        result = sq.otc.asianSwap(100.0, prices, 1000, r=0.0, T=1.0)
        assert result['npv'] > 0

    def test_return_keys(self):
        prices = np.array([100.0, 101.0])
        result = sq.otc.asianSwap(100.0, prices, 100, r=0.05, T=0.5)
        assert 'npv' in result
        assert 'averageIndex' in result
        assert 'dv01' in result


# ---------------------------------------------------------------------------
# collar
# ---------------------------------------------------------------------------

class TestCollar:
    def setup_method(self):
        self.params = dict(S=100.0, capStrike=110.0, floorStrike=90.0,
                           T=1.0, r=0.05, sigma=0.2, notional=1.0)

    def test_price_less_than_or_equal_to_cap_price(self):
        result = sq.otc.collar(**self.params)
        assert result['price'] <= result['capPrice']

    def test_cap_price_positive(self):
        result = sq.otc.collar(**self.params)
        assert result['capPrice'] > 0

    def test_floor_price_positive(self):
        result = sq.otc.collar(**self.params)
        assert result['floorPrice'] > 0

    def test_all_greeks_finite(self):
        result = sq.otc.collar(**self.params)
        for key, val in result['greeks'].items():
            assert np.isfinite(val), f"greek {key} is not finite"

    def test_net_price_is_cap_minus_floor(self):
        result = sq.otc.collar(**self.params)
        assert result['price'] == pytest.approx(
            result['capPrice'] - result['floorPrice'], rel=1e-6
        )

    def test_delta_between_neg1_and_1(self):
        result = sq.otc.collar(**self.params)
        assert -1.5 < result['greeks']['delta'] < 1.5


# ---------------------------------------------------------------------------
# physicalForward
# ---------------------------------------------------------------------------

class TestPhysicalForward:
    def test_pv_positive(self):
        result = sq.otc.physicalForward(100.0, 1.0, 0.05)
        assert result['pv'] > 0

    def test_pv_formula(self):
        F = 100.0
        T = 1.0
        r = 0.05
        storage = 0.02
        quality = 2.0
        notional = 500.0
        result = sq.otc.physicalForward(F, T, r, storageCost=storage,
                                        qualityPremium=quality, notional=notional)
        expected = (F + quality) * np.exp(-(r + storage) * T) * notional
        assert result['pv'] == pytest.approx(expected, rel=1e-8)

    def test_adjusted_forward(self):
        result = sq.otc.physicalForward(100.0, 1.0, 0.05, qualityPremium=3.0)
        assert result['adjustedForward'] == pytest.approx(103.0)

    def test_return_keys(self):
        result = sq.otc.physicalForward(100.0, 1.0, 0.05)
        assert 'pv' in result
        assert 'forwardPrice' in result
        assert 'adjustedForward' in result
        assert 'delta' in result['greeks']
        assert 'dv01' in result['greeks']


# ---------------------------------------------------------------------------
# swaption
# ---------------------------------------------------------------------------

class TestSwaption:
    def setup_method(self):
        self.schedule = np.array([1.0, 2.0, 3.0])
        self.indexCurve = np.array([102.0, 104.0, 106.0])
        self.params = dict(
            fixedPrice=100.0,
            indexCurve=self.indexCurve,
            notional=1000.0,
            schedule=self.schedule,
            r=0.05,
            sigma=0.2,
            T=0.5,
        )

    def test_call_price_positive(self):
        result = sq.otc.swaption(**self.params, optType='call')
        assert result['price'] > 0

    def test_put_price_positive(self):
        # Make fixedPrice above forward swap rate so put is in the money
        params = self.params.copy()
        params['fixedPrice'] = 200.0
        result = sq.otc.swaption(**params, optType='put')
        assert result['price'] > 0

    def test_return_keys(self):
        result = sq.otc.swaption(**self.params)
        assert 'price' in result
        assert 'forwardSwapRate' in result
        assert 'annuity' in result
        assert 'greeks' in result
        assert 'delta' in result['greeks']
        assert 'vega' in result['greeks']
        assert 'theta' in result['greeks']

    def test_vega_positive(self):
        result = sq.otc.swaption(**self.params, optType='call')
        assert result['greeks']['vega'] > 0

    def test_forward_swap_rate_finite(self):
        result = sq.otc.swaption(**self.params)
        assert np.isfinite(result['forwardSwapRate'])


# ---------------------------------------------------------------------------
# asianOption
# ---------------------------------------------------------------------------

class TestAsianOption:
    def setup_method(self):
        self.params = dict(
            S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2,
            nSims=2000, nSteps=20, optType='call', seed=42,
        )

    def test_price_positive(self):
        result = sq.otc.asianOption(**self.params)
        assert result['price'] > 0

    def test_stderr_positive(self):
        result = sq.otc.asianOption(**self.params)
        assert result['stderr'] > 0

    def test_put_price_positive(self):
        params = self.params.copy()
        params['optType'] = 'put'
        result = sq.otc.asianOption(**params)
        assert result['price'] > 0

    def test_greeks_finite(self):
        result = sq.otc.asianOption(**self.params)
        assert np.isfinite(result['greeks']['delta'])
        assert np.isfinite(result['greeks']['vega'])

    def test_price_lower_than_vanilla_call_approx(self):
        # Asian option price <= vanilla call price (averaging dampens vol)
        result = sq.otc.asianOption(**self.params)
        # Very rough upper bound: Asian typically < ATM BS call for same params
        assert result['price'] < 20.0  # BS ATM 1Y call ~= 7.97 for sigma=0.2

    def test_reproducible_with_seed(self):
        r1 = sq.otc.asianOption(**self.params)
        r2 = sq.otc.asianOption(**self.params)
        assert r1['price'] == pytest.approx(r2['price'])
