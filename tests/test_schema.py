"""
Tests for sipQuant.schema — data contract layer.
"""
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import sipQuant as sq


class TestPriceSeries:
    def test_basic(self):
        ps = sq.schema.PriceSeries([1, 2, 3], [10.0, 11.0, 12.0], 'broker_a', 'hay')
        assert ps['type'] == 'PriceSeries'
        assert ps['n'] == 3
        assert ps['market'] == 'hay'

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sq.schema.PriceSeries([], [], 'src', 'mkt')

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            sq.schema.PriceSeries([1, 2], [10.0], 'src', 'mkt')

    def test_values_are_float(self):
        ps = sq.schema.PriceSeries([1], [10], 'src', 'mkt')
        assert ps['values'].dtype == float

    def test_grade_optional(self):
        ps = sq.schema.PriceSeries([1], [10.0], 'src', 'mkt')
        assert ps['grade'] is None


class TestSparsePriceSeries:
    def test_basic(self):
        dates = np.array(['2026-01-01', '2026-01-08', '2026-01-15'], dtype='datetime64')
        sps = sq.schema.SparsePriceSeries(dates, [10.0, 11.0, 12.0], 'src', 'mkt', maxGapDays=10)
        assert sps['type'] == 'SparsePriceSeries'
        assert sps['n'] == 3

    def test_gap_flags_computed(self):
        dates = np.array(['2026-01-01', '2026-01-15'], dtype='datetime64')
        sps = sq.schema.SparsePriceSeries(dates, [10.0, 11.0], 'src', 'mkt', maxGapDays=7)
        assert sps['gapFlags'] is not None
        assert sps['gapFlags'][0] is True or sps['gapFlags'][0] == True

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sq.schema.SparsePriceSeries([], [], 'src', 'mkt')


class TestTradeRecord:
    def test_basic(self):
        tr = sq.schema.TradeRecord('2026-01-01', 100.0, 500.0, 'premium', 'lethbridge', 'calgary', 'CP001')
        assert tr['type'] == 'TradeRecord'
        assert tr['price'] == 100.0

    def test_negative_price_raises(self):
        with pytest.raises(ValueError):
            sq.schema.TradeRecord('2026-01-01', -10.0, 500.0, 'grade', 'A', 'B', 'CP001')

    def test_zero_volume_raises(self):
        with pytest.raises(ValueError):
            sq.schema.TradeRecord('2026-01-01', 100.0, 0.0, 'grade', 'A', 'B', 'CP001')


class TestQuoteSheet:
    def test_basic(self):
        qs = sq.schema.QuoteSheet('2026-01-01', 99.0, 101.0, 100.0, 'broker', 'hay', 'premium', 0.25)
        assert qs['type'] == 'QuoteSheet'
        assert qs['mid'] == 100.0

    def test_bid_greater_than_ask_raises(self):
        with pytest.raises(ValueError):
            sq.schema.QuoteSheet('2026-01-01', 102.0, 100.0, 101.0, 'broker', 'hay', 'prem', 0.25)

    def test_negative_tenor_raises(self):
        with pytest.raises(ValueError):
            sq.schema.QuoteSheet('2026-01-01', 99.0, 101.0, 100.0, 'broker', 'hay', 'prem', -0.1)


class TestForwardCurve:
    def test_basic(self):
        fc = sq.schema.ForwardCurve([0.25, 0.5, 1.0], [100.0, 102.0, 105.0], '2026-01-01', 'hay')
        assert fc['type'] == 'ForwardCurve'
        assert fc['n'] == 3

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            sq.schema.ForwardCurve([], [], '2026-01-01', 'hay')

    def test_non_monotone_tenors_raises(self):
        with pytest.raises(ValueError):
            sq.schema.ForwardCurve([1.0, 0.5], [100.0, 102.0], '2026-01-01', 'hay')

    def test_non_positive_prices_raises(self):
        with pytest.raises(ValueError):
            sq.schema.ForwardCurve([0.25, 0.5], [100.0, -5.0], '2026-01-01', 'hay')


class TestOTCPosition:
    def test_basic(self):
        pos = sq.schema.OTCPosition('commodity_swap', 'pay_fixed', 1000.0, 190.0, '2026-12-31', 'CP001')
        assert pos['type'] == 'OTCPosition'
        assert pos['greeks']['delta'] == 0.0

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            sq.schema.OTCPosition('swap', 'unknown_dir', 1000.0, 190.0, '2026-12-31', 'CP001')

    def test_zero_notional_raises(self):
        with pytest.raises(ValueError):
            sq.schema.OTCPosition('swap', 'buy', 0.0, 190.0, '2026-12-31', 'CP001')

    def test_custom_greeks(self):
        pos = sq.schema.OTCPosition('collar', 'long', 500.0, 195.0, '2026-06-30', 'CP002',
                                     greeks={'delta': 0.5, 'gamma': 0.02, 'vega': 3.0, 'theta': -0.1, 'rho': 0.1})
        assert pos['greeks']['delta'] == 0.5


class TestIndexSpec:
    def test_basic(self):
        spec = sq.schema.IndexSpec('SIP-AHI-001', '1.0', ['hay_premium', 'hay_std'],
                                    'volume', 'monthly_last_business_day', '2026-01-01')
        assert spec['type'] == 'IndexSpec'
        assert spec['name'] == 'SIP-AHI-001'

    def test_invalid_weights_method_raises(self):
        with pytest.raises(ValueError):
            sq.schema.IndexSpec('SIP-AHI-001', '1.0', ['hay'], 'invalid_method', 'monthly', '2026-01-01')

    def test_empty_constituents_raises(self):
        with pytest.raises(ValueError):
            sq.schema.IndexSpec('SIP-AHI-001', '1.0', [], 'equal', 'monthly', '2026-01-01')


class TestValidate:
    def test_valid_price_series(self):
        ps = sq.schema.PriceSeries([1, 2, 3], [10.0, 11.0, 12.0], 'src', 'mkt')
        assert sq.schema.validate(ps) == []

    def test_non_dict_returns_error(self):
        errors = sq.schema.validate("not a dict")
        assert len(errors) > 0

    def test_missing_type_returns_error(self):
        errors = sq.schema.validate({'values': [1, 2]})
        assert any("type" in e.lower() for e in errors)

    def test_unknown_type_returns_error(self):
        errors = sq.schema.validate({'type': 'UnknownType'})
        assert len(errors) > 0

    def test_valid_trade_record(self):
        tr = sq.schema.TradeRecord('2026-01-01', 100.0, 500.0, 'premium', 'A', 'B', 'CP001')
        assert sq.schema.validate(tr) == []

    def test_valid_forward_curve(self):
        fc = sq.schema.ForwardCurve([0.25, 0.5], [100.0, 102.0], '2026-01-01', 'hay')
        assert sq.schema.validate(fc) == []

    def test_valid_index_spec(self):
        spec = sq.schema.IndexSpec('SIP-AHI-001', '1.0', ['hay'], 'equal', 'monthly', '2026-01-01')
        assert sq.schema.validate(spec) == []
