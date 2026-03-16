# sipQuant commodity — SIP Global Workflow Guide

## Purpose

The `commodity` module provides physical commodity pricing primitives used throughout
the SIP index and OTC stack: seasonality decomposition, convenience yield extraction,
basis calculation, grade adjustment, and local forward curve construction.

---

## Function Reference

| Function | Use Case |
|---|---|
| `seasonality` | Decompose index series into trend + seasonal + residual |
| `convenienceYield` | Extract convenience yield from spot/futures pair |
| `basis` | Cash price minus reference (local vs. benchmark) |
| `gradeAdjustment` | Apply moisture, dockage, or quality adjustments |
| `transportDifferential` | Delivered price at destination |
| `localForwardCurve` | Build cluster-specific forward curve |
| `rollingRollCost` | Estimate carry cost from forward curve shape |

---

## Step-by-Step: Local Forward Curve for a Commodity Cluster

### 1. Decompose price series to understand seasonality

```python
import sipQuant as sq
import numpy as np

# Alberta Hay Premium — weekly observations (52 per year)
ps = sq.schema.PriceSeries(dates, prices, 'broker_a', 'alberta_hay_premium')

decomp = sq.commodity.seasonality(
    dates=np.arange(len(ps['values'])),
    values=ps['values'],
    period=52,        # 52-week annual cycle
    method='stl'
)

trend    = decomp['trend']
seasonal = decomp['seasonal']
residual = decomp['residual']

# Seasonal index: use for forward curve seasonal adjustment
```

### 2. Extract convenience yield from nearby futures

```python
# Spot and 3-month futures prices
spotPrice   = 187.50
futuresPrice = 192.00
tenor       = 0.25     # 3 months
r           = 0.046
storageCost = 0.02     # 2% per year (bale storage, shrinkage)

cy = sq.commodity.convenienceYield(spotPrice, futuresPrice, tenor, r, storageCost)
print(f"Convenience yield: {cy['convenienceYield']:.4f}")
print(f"Net carry: {cy['netCarry']:.4f}")
```

### 3. Compute local basis

```python
# Alberta local cash vs. the SIP-AHI composite benchmark
cash_price = 185.00
benchmark  = 188.00

b = sq.commodity.basis(cash_price, benchmark,
                        market='lethbridge_ab',
                        grade='premium_bale_14pct_moisture')
print(f"Basis: {b['basis']:.2f} $/tonne ({b['basisBps']:.0f} bps)")
```

### 4. Apply grade adjustment to base price

```python
# Grade differentials from current methodology spec
grade_factors = {
    'moisture_premium_14pct': +2.50,
    'dockage_discount_2pct' : -1.80,
    'test_weight_premium'   : +1.20,
}

adj = sq.commodity.gradeAdjustment(basePrice=185.0, gradeFactors=grade_factors)
print(f"Adjusted price: {adj['adjustedPrice']:.2f}")
print(f"Total adjustment: {adj['totalAdjustment']:+.2f}")
```

### 5. Delivered price including logistics

```python
origin_price = 185.0
freight      = 8.50    # $/tonne truck freight Lethbridge → Calgary
handling     = 1.20    # feedlot receiving fee
insurance    = 0.30

delivered = sq.commodity.transportDifferential(origin_price, freight, handling, insurance)
print(f"Delivered price: {delivered['deliveredPrice']:.2f}")
```

### 6. Build local forward curve

```python
tenors = np.array([0.0, 0.25, 0.50, 0.75, 1.0])

fwd = sq.commodity.localForwardCurve(
    spotPrice       = 187.50,
    tenor           = tenors,
    r               = 0.046,
    convYield       = 0.028,    # from step 2
    storageCost     = 0.020,
    basisAdjustment = -2.50     # local basis
)

# Wrap in schema object for OTC pricing
curve = sq.schema.ForwardCurve(
    tenors   = fwd['tenors'],
    prices   = fwd['forwards'],
    baseDate = '2026-03-14',
    market   = 'alberta_hay_premium'
)
```

### 7. Estimate roll costs for index methodology

```python
rollDates = np.array([0, 1, 2, 3])  # quarterly roll indices
roll = sq.commodity.rollingRollCost(fwd, rollDates)
print(f"Quarterly roll costs: {roll['rollCosts']}")
print(f"Annualized roll cost: {roll['annualizedRollCost']:.4f}")
```

---

## Adapting to Any Cluster

Replace spot prices, storage costs, convenience yields, and grade factors with
the target cluster's market data. All functions are commodity-agnostic.
Clusters with strong seasonality (hay, silage, straw) benefit most from step 1.
Clusters with liquid futures (canola, wheat) benefit most from step 2.
