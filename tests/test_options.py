"""
Tests for sipQuant.options — option pricing models.
"""
import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sipQuant as sq


class TestBlackScholes:
    def test_call_price_positive(self):
        res = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2)
        assert res['price'] > 0

    def test_put_price_positive(self):
        res = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2, optType='put')
        assert res['price'] > 0

    def test_put_call_parity(self):
        S, K, T, r, sigma, q = 100, 100, 1.0, 0.05, 0.2, 0.02
        call = sq.options.blackScholes(S, K, T, r, sigma, q, 'call')['price']
        put  = sq.options.blackScholes(S, K, T, r, sigma, q, 'put')['price']
        parity = S * math.exp(-q * T) - K * math.exp(-r * T)
        assert abs((call - put) - parity) < 1e-6

    def test_call_delta_in_range(self):
        res = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2)
        assert 0.0 < res['delta'] < 1.0

    def test_put_delta_in_range(self):
        res = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2, optType='put')
        assert -1.0 < res['delta'] < 0.0

    def test_gamma_positive(self):
        res = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2)
        assert res['gamma'] > 0

    def test_vega_positive(self):
        res = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2)
        assert res['vega'] > 0

    def test_call_increases_with_spot(self):
        p1 = sq.options.blackScholes(95, 100, 1.0, 0.05, 0.2)['price']
        p2 = sq.options.blackScholes(105, 100, 1.0, 0.05, 0.2)['price']
        assert p2 > p1

    def test_zero_time_intrinsic_only(self):
        res = sq.options.blackScholes(110, 100, 0.0, 0.05, 0.2, optType='call')
        assert abs(res['price'] - 10.0) < 1e-6

    def test_deep_itm_call_delta_near_one(self):
        res = sq.options.blackScholes(200, 100, 1.0, 0.05, 0.2)
        assert res['delta'] > 0.95

    def test_known_price(self):
        # Classic result: BS(100,100,1,0.05,0.2) ≈ 10.45 call
        res = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2, q=0.0, optType='call')
        assert abs(res['price'] - 10.45) < 0.1


class TestSpread:
    def test_call_price_positive(self):
        res = sq.options.spread(110, 100, 5, 1.0, 0.05, 0.25, 0.20, 0.7)
        assert res['price'] > 0

    def test_put_price_non_negative(self):
        res = sq.options.spread(110, 100, 5, 1.0, 0.05, 0.25, 0.20, 0.7, optType='put')
        assert res['price'] >= 0

    def test_delta1_positive_for_call(self):
        res = sq.options.spread(110, 100, 5, 1.0, 0.05, 0.25, 0.20, 0.7)
        assert res['delta1'] > 0

    def test_delta2_negative_for_call(self):
        res = sq.options.spread(110, 100, 5, 1.0, 0.05, 0.25, 0.20, 0.7)
        assert res['delta2'] < 0

    def test_returns_all_keys(self):
        res = sq.options.spread(110, 100, 5, 1.0, 0.05, 0.25, 0.20, 0.7)
        for k in ('price', 'delta1', 'delta2', 'gamma1', 'gamma2', 'vega1', 'vega2'):
            assert k in res


class TestBarrier:
    def test_down_and_out_below_vanilla(self):
        van = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2)['price']
        bar = sq.options.barrier(100, 100, 1.0, 0.05, 0.2, 80, optType='call',
                                  barrierType='down-and-out')['price']
        assert bar <= van + 1e-6

    def test_in_plus_out_equals_vanilla(self):
        S, K, T, r, sig, H = 100, 100, 1.0, 0.05, 0.2, 80
        out = sq.options.barrier(S, K, T, r, sig, H, optType='call',
                                  barrierType='down-and-out')['price']
        inn = sq.options.barrier(S, K, T, r, sig, H, optType='call',
                                  barrierType='down-and-in')['price']
        van = sq.options.blackScholes(S, K, T, r, sig, optType='call')['price']
        assert abs(out + inn - van) < 0.05

    def test_delta_finite(self):
        res = sq.options.barrier(100, 100, 1.0, 0.05, 0.2, 80)
        assert math.isfinite(res['delta'])

    def test_gamma_finite(self):
        res = sq.options.barrier(100, 100, 1.0, 0.05, 0.2, 80)
        assert math.isfinite(res['gamma'])


class TestAsian:
    def test_geometric_price_positive(self):
        res = sq.options.asian(100, 100, 1.0, 0.05, 0.2, avgType='geometric')
        assert res['price'] > 0

    def test_geometric_below_vanilla(self):
        geo = sq.options.asian(100, 100, 1.0, 0.05, 0.2, avgType='geometric')['price']
        van = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2)['price']
        assert geo <= van + 1e-6

    def test_arithmetic_price_positive(self):
        res = sq.options.asian(100, 100, 1.0, 0.05, 0.2, avgType='arithmetic',
                                nSims=2000, nSteps=50)
        assert res['price'] > 0

    def test_arithmetic_has_stderr(self):
        res = sq.options.asian(100, 100, 1.0, 0.05, 0.2, avgType='arithmetic',
                                nSims=2000, nSteps=50)
        assert 'stderr' in res
        assert res['stderr'] > 0

    def test_geometric_returns_greeks(self):
        res = sq.options.asian(100, 100, 1.0, 0.05, 0.2, avgType='geometric')
        for k in ('price', 'delta', 'gamma', 'vega', 'rho', 'theta'):
            assert k in res


class TestBinomial:
    def test_european_call_close_to_bs(self):
        bs  = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2)['price']
        bin_p = sq.options.binomial(100, 100, 1.0, 0.05, 0.2, N=200)['price']
        assert abs(bin_p - bs) < 0.5

    def test_american_put_geq_european(self):
        eur = sq.options.binomial(100, 100, 1.0, 0.05, 0.2, optType='put',
                                   N=100, american=False)['price']
        amr = sq.options.binomial(100, 100, 1.0, 0.05, 0.2, optType='put',
                                   N=100, american=True)['price']
        assert amr >= eur - 1e-8

    def test_returns_keys(self):
        res = sq.options.binomial(100, 100, 1.0, 0.05, 0.2)
        assert 'price' in res and 'delta' in res


class TestTrinomial:
    def test_call_close_to_bs(self):
        bs  = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2)['price']
        tri = sq.options.trinomial(100, 100, 1.0, 0.05, 0.2, N=100)['price']
        assert abs(tri - bs) < 0.5

    def test_returns_delta_gamma(self):
        res = sq.options.trinomial(100, 100, 1.0, 0.05, 0.2)
        assert math.isfinite(res['delta'])
        assert math.isfinite(res['gamma'])


class TestMonteCarlo:
    def test_price_close_to_bs(self):
        bs  = sq.options.blackScholes(100, 100, 1.0, 0.05, 0.2)['price']
        mc  = sq.options.monteCarlo(100, 100, 1.0, 0.05, 0.2, nSims=20000, seed=0)
        assert abs(mc['price'] - bs) < mc['stderr'] * 3 + 0.3

    def test_returns_all_keys(self):
        res = sq.options.monteCarlo(100, 100, 1.0, 0.05, 0.2, nSims=1000, seed=1)
        for k in ('price', 'stderr', 'delta', 'gamma', 'vega', 'rho', 'theta'):
            assert k in res

    def test_stderr_positive(self):
        res = sq.options.monteCarlo(100, 100, 1.0, 0.05, 0.2, nSims=1000, seed=2)
        assert res['stderr'] > 0


class TestImpliedVol:
    def test_round_trip(self):
        sigma = 0.25
        price = sq.options.blackScholes(100, 100, 1.0, 0.05, sigma)['price']
        iv    = sq.options.impliedVol(price, 100, 100, 1.0, 0.05)
        assert abs(iv - sigma) < 1e-4

    def test_put_round_trip(self):
        sigma = 0.18
        price = sq.options.blackScholes(100, 110, 0.5, 0.04, sigma, optType='put')['price']
        iv    = sq.options.impliedVol(price, 100, 110, 0.5, 0.04, optType='put')
        assert abs(iv - sigma) < 1e-4

    def test_impossible_price_returns_nan(self):
        iv = sq.options.impliedVol(1e6, 100, 100, 1.0, 0.05)
        assert math.isnan(iv)
