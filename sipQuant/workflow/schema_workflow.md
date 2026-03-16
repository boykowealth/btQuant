# sipQuant schema — SIP Global Workflow Guide

## Purpose

The `schema` module is the data contract layer. Every function in sipQuant accepts
schema objects rather than raw arrays. This ensures validated, consistent inputs
across the pricing, risk, and index stack.

---

## Schema Objects

| Object | Use |
|---|---|
| `PriceSeries` | Regular daily/weekly price series (broker quotes, exchange settlements) |
| `SparsePriceSeries` | Irregular physical trade observations (hay, grain, oilseed thin markets) |
| `TradeRecord` | Single physical trade (price, volume, grade, origin, destination) |
| `QuoteSheet` | OTC broker quote with bid/ask/mid and tenor |
| `ForwardCurve` | Forward curve for pricing and hedging |
| `OTCPosition` | Live OTC position for book aggregation |
| `IndexSpec` | Immutable index methodology specification |

---

## Step-by-Step: Onboarding a New Commodity Cluster

This workflow applies to any SIP cluster (Alberta Hay, Western Canadian Feed Wheat,
Prairie Canola Meal, BC Silage Corn, etc.).

### 1. Ingest trade data as TradeRecords

```python
import sipQuant as sq

trade = sq.schema.TradeRecord(
    date='2026-03-14',
    price=187.50,
    volume=500.0,
    grade='premium_bale_14pct_moisture',
    origin='lethbridge_ab',
    destination='red_deer_ab',
    counterpartyId='CP_ANON_004'
)

errors = sq.schema.validate(trade)
assert errors == [], f"Invalid trade: {errors}"
```

### 2. Build a price series from broker quotes

```python
import numpy as np

dates = np.array(['2026-01-06', '2026-01-13', '2026-01-20'], dtype='datetime64')
prices = np.array([182.0, 184.5, 187.0])

ps = sq.schema.PriceSeries(
    dates=dates,
    values=prices,
    source='broker_prairie_ag',
    market='alberta_hay_premium',
    grade='premium_bale_14pct_moisture'
)
```

### 3. Define the index specification

```python
spec = sq.schema.IndexSpec(
    name='SIP-AHI-001',
    version='1.0',
    constituents=['alberta_hay_premium', 'alberta_hay_standard'],
    weightsMethod='volume',
    rollRule='monthly_last_business_day',
    effectiveDate='2026-01-01'
)
```

### 4. Build a forward curve for OTC pricing

```python
curve = sq.schema.ForwardCurve(
    tenors=np.array([0.25, 0.5, 0.75, 1.0]),
    prices=np.array([188.0, 190.5, 192.0, 193.5]),
    baseDate='2026-03-14',
    market='alberta_hay_premium',
    methodology='linear'
)
```

### 5. Create an OTC position record

```python
pos = sq.schema.OTCPosition(
    instrumentType='commodity_swap',
    direction='pay_fixed',
    notional=1000.0,
    strikeOrFixed=190.0,
    expiry='2026-12-31',
    counterpartyId='CP_ANON_004',
    greeks={'delta': 0.95, 'gamma': 0.0, 'vega': 2.5, 'theta': -0.05, 'rho': 0.3}
)
```

---

## Validation Pattern

Always validate before passing to pricing or index functions:

```python
for obj in [trade, ps, spec, curve, pos]:
    errors = sq.schema.validate(obj)
    if errors:
        raise ValueError(f"Schema validation failed: {errors}")
```

---

## Adapting to Any Cluster

Replace `market`, `grade`, `source`, and `constituents` with the target cluster's
identifiers as defined in `SIP_Cluster_Taxonomy.md`. All downstream functions
(options, otc, index, book) accept the same schema objects regardless of cluster.
