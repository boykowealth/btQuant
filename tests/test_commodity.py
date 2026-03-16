"""
Tests for sipQuant.commodity
SIP Global (Systematic Index Partners)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import sipQuant as sq


# ---------------------------------------------------------------------------
# seasonality
# ---------------------------------------------------------------------------

class TestSeasonality:
    def setup_method(self):
        rng = np.random.default_rng(0)
        n = 104
        self.dates = np.arange(n)
        trend = np.linspace(100, 120, n)
        seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 52)
        noise = rng.normal(0, 0.5, n)
        self.values = trend + seasonal + noise

    def test_reconstruction(self):
        result = sq.commodity.seasonality(self.dates, self.values, period=52)
        reconstructed = result['trend'] + result['seasonal'] + result['residual']
        np.testing.assert_allclose(reconstructed, result['values'], rtol=1e-10)

    def test_shapes_match(self):
        result = sq.commodity.seasonality(self.dates, self.values, period=52)
        n = len(self.values)
        assert result['trend'].shape == (n,)
        assert result['seasonal'].shape == (n,)
        assert result['residual'].shape == (n,)
        assert result['values'].shape == (n,)

    def test_period_stored(self):
        result = sq.commodity.seasonality(self.dates, self.values, period=52)
        assert result['period'] == 52

    def test_method_stored(self):
        result = sq.commodity.seasonality(self.dates, self.values, method='additive')
        assert result['method'] == 'additive'


# ---------------------------------------------------------------------------
# convenienceYield
# ---------------------------------------------------------------------------

class TestConvenienceYield:
    def test_zero_cy_when_forward_equals_carry(self):
        S = 100.0
        r = 0.05
        t = 1.0
        storage = 0.02
        # F = S * exp((r + storage) * t) => cy = 0
        F = S * np.exp((r + storage) * t)
        result = sq.commodity.convenienceYield(S, F, t, r, storageCost=storage)
        assert abs(result['convenienceYield']) < 1e-10

    def test_positive_cy_when_futures_below_full_carry(self):
        S = 100.0
        r = 0.05
        t = 1.0
        F = 103.0  # below full carry S*exp(r*t) ~ 105.13
        result = sq.commodity.convenienceYield(S, F, t, r)
        assert result['convenienceYield'] > 0

    def test_return_keys(self):
        result = sq.commodity.convenienceYield(100, 102, 1.0, 0.05)
        assert 'convenienceYield' in result
        assert 'carryAdjustedForward' in result
        assert 'netCarry' in result

    def test_carry_adjusted_forward_equals_futures(self):
        S = 100.0
        r = 0.05
        t = 1.0
        F = 102.0
        result = sq.commodity.convenienceYield(S, F, t, r)
        np.testing.assert_allclose(result['carryAdjustedForward'], F, rtol=1e-10)


# ---------------------------------------------------------------------------
# basis
# ---------------------------------------------------------------------------

class TestBasis:
    def test_basis_equals_cash_minus_reference(self):
        result = sq.commodity.basis(105.0, 100.0)
        assert result['basis'] == pytest.approx(5.0)

    def test_negative_basis(self):
        result = sq.commodity.basis(98.0, 100.0)
        assert result['basis'] == pytest.approx(-2.0)

    def test_basis_bps_sign_matches_basis(self):
        result = sq.commodity.basis(105.0, 100.0)
        assert result['basisBps'] > 0

    def test_market_grade_stored(self):
        result = sq.commodity.basis(100.0, 100.0, market='alberta_hay', grade='premium')
        assert result['market'] == 'alberta_hay'
        assert result['grade'] == 'premium'

    def test_zero_basis(self):
        result = sq.commodity.basis(100.0, 100.0)
        assert result['basis'] == pytest.approx(0.0)
        assert result['basisBps'] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# gradeAdjustment
# ---------------------------------------------------------------------------

class TestGradeAdjustment:
    def test_positive_factors_increase_price(self):
        result = sq.commodity.gradeAdjustment(100.0, [1.5, 0.5])
        assert result['adjustedPrice'] > 100.0

    def test_negative_factors_decrease_price(self):
        result = sq.commodity.gradeAdjustment(100.0, [-2.0, -1.0])
        assert result['adjustedPrice'] < 100.0

    def test_dict_grade_factors(self):
        factors = {'proteinBonus': 1.0, 'moisturePenalty': -0.5}
        result = sq.commodity.gradeAdjustment(100.0, factors)
        assert result['adjustedPrice'] == pytest.approx(100.5)
        assert result['totalAdjustment'] == pytest.approx(0.5)

    def test_adjusted_price_formula(self):
        result = sq.commodity.gradeAdjustment(50.0, [3.0, 2.0])
        assert result['adjustedPrice'] == pytest.approx(55.0)
        assert result['totalAdjustment'] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# transportDifferential
# ---------------------------------------------------------------------------

class TestTransportDifferential:
    def test_delivered_price_sum(self):
        result = sq.commodity.transportDifferential(
            originPrice=100.0,
            freightCost=5.0,
            handlingCost=1.0,
            insuranceCost=0.5,
        )
        assert result['deliveredPrice'] == pytest.approx(106.5)

    def test_freight_only(self):
        result = sq.commodity.transportDifferential(100.0, 8.0)
        assert result['deliveredPrice'] == pytest.approx(108.0)
        assert result['totalLogisticsCost'] == pytest.approx(8.0)

    def test_breakdown_keys(self):
        result = sq.commodity.transportDifferential(100.0, 5.0, 1.0, 0.5)
        assert 'freight' in result['breakdown']
        assert 'handling' in result['breakdown']
        assert 'insurance' in result['breakdown']

    def test_origin_price_preserved(self):
        result = sq.commodity.transportDifferential(100.0, 5.0)
        assert result['originPrice'] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# localForwardCurve
# ---------------------------------------------------------------------------

class TestLocalForwardCurve:
    def test_forwards_positive(self):
        tenors = np.array([0.25, 0.5, 1.0, 2.0])
        result = sq.commodity.localForwardCurve(100.0, tenors, r=0.05, convYield=0.02)
        assert np.all(result['forwards'] > 0)

    def test_shape_matches_tenor(self):
        tenors = np.linspace(0.25, 2.0, 8)
        result = sq.commodity.localForwardCurve(100.0, tenors, r=0.05, convYield=0.02)
        assert result['forwards'].shape == tenors.shape

    def test_net_carry_formula(self):
        result = sq.commodity.localForwardCurve(
            100.0, np.array([1.0]), r=0.05, convYield=0.02, storageCost=0.01
        )
        assert result['netCarry'] == pytest.approx(0.05 + 0.01 - 0.02)

    def test_zero_tenor_at_spot(self):
        tenors = np.array([0.0, 1.0])
        result = sq.commodity.localForwardCurve(100.0, tenors, r=0.05, convYield=0.05)
        # At t=0, F = spot (no carry)
        assert result['forwards'][0] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# rollingRollCost
# ---------------------------------------------------------------------------

class TestRollingRollCost:
    def setup_method(self):
        tenors = np.linspace(0.25, 2.0, 8)
        forwards = 100.0 * np.exp(0.03 * tenors)
        self.forwardCurve = {'tenors': tenors, 'forwards': forwards}

    def test_roll_costs_length(self):
        rollDates = np.array([0, 2, 4])
        result = sq.commodity.rollingRollCost(self.forwardCurve, rollDates)
        assert len(result['rollCosts']) == 3

    def test_total_roll_cost_is_sum(self):
        rollDates = np.array([0, 2, 4])
        result = sq.commodity.rollingRollCost(self.forwardCurve, rollDates)
        assert result['totalRollCost'] == pytest.approx(np.sum(result['rollCosts']))

    def test_positive_carry_gives_positive_roll_cost(self):
        rollDates = np.array([0, 1, 2])
        result = sq.commodity.rollingRollCost(self.forwardCurve, rollDates)
        # In contango, each roll incurs positive cost (buy higher tenor)
        assert np.all(result['rollCosts'] > 0)

    def test_prices_key_fallback(self):
        curve = {'tenors': self.forwardCurve['tenors'], 'prices': self.forwardCurve['forwards']}
        rollDates = np.array([0, 1])
        result = sq.commodity.rollingRollCost(curve, rollDates)
        assert len(result['rollCosts']) == 2
