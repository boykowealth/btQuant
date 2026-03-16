# sipQuant bootstrap — SIP Global Workflow Guide

## Purpose

The `bootstrap` module constructs forward curves, discount curves, and vol surfaces
from market observable prices. Outputs are the primary inputs to OTC pricing in `otc.py`.

---

## Step-by-Step: Building a Local Forward Curve

### 1. Spot price + futures quotes → forward curve

```python
import sipQuant as sq
import numpy as np

# Alberta Hay Premium: spot + 4 quarterly futures
tenors = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
rates = np.array([0.045, 0.045, 0.046, 0.046, 0.047])
storageCosts = np.array([0.02, 0.02, 0.02, 0.02, 0.02])
convenienceYields = np.array([0.03, 0.028, 0.025, 0.022, 0.020])

curve = sq.bootstrap.forwardCurve(
    spotPrice=187.50,
    tenors=tenors,
    rates=rates,
    storageCosts=storageCosts,
    convenienceYields=convenienceYields
)
# curve['forwards']: forward price array at each tenor
```

### 2. Bootstrap discount curve from zero rates

```python
discCurve = sq.bootstrap.discountCurve(
    tenors=np.array([0.25, 0.5, 1.0, 2.0, 3.0]),
    rates=np.array([0.044, 0.045, 0.047, 0.050, 0.052]),
    method='log_linear'
)
# discCurve['discountFactors']: DF at each tenor
```

### 3. Bootstrap convenience yield curve from observed futures

```python
convYieldCurve = sq.bootstrap.convenienceYieldCurve(
    spotPrice=187.50,
    futuresPrices=np.array([189.0, 191.5, 193.0]),
    tenors=np.array([0.25, 0.5, 1.0]),
    rates=np.array([0.045, 0.046, 0.047])
)
# convYieldCurve['convenienceYields']: y(t) for each tenor
```

### 4. Implied vol surface construction

```python
strikes = np.array([160, 175, 185, 195, 205, 220])
tenors = np.array([0.25, 0.5, 1.0])
vols = np.array([
    [0.25, 0.22, 0.20, 0.22, 0.25, 0.30],
    [0.23, 0.21, 0.19, 0.21, 0.24, 0.28],
    [0.22, 0.20, 0.18, 0.20, 0.22, 0.26],
])

surface = sq.bootstrap.volSurface(strikes, tenors, vols)

# Query vol at any strike/tenor
vol = sq.bootstrap.interpVol(surface, strike=190.0, tenor=0.75)
```

### 5. Add basis spread for local market

```python
adjustedCurve = sq.bootstrap.spreadCurve(
    baseRates=discCurve['zeroRates'],
    spreadBps=np.array([15.0, 15.0, 18.0, 20.0, 22.0]),
    tenors=discCurve['tenors']
)
```

---

## Pipeline: Curve → OTC Pricing

```python
# Build inputs
fwdCurve = sq.schema.ForwardCurve(
    tenors=curve['tenors'],
    prices=curve['forwards'],
    baseDate='2026-03-14',
    market='alberta_hay_premium'
)

# Feed into OTC swap pricing
swap = sq.otc.commoditySwap(
    fixedPrice=190.0,
    indexCurve=fwdCurve['prices'],
    notional=500.0,
    schedule=fwdCurve['tenors'],
    r=0.046
)
```

---

## Adapting to Any Cluster

Replace spot price, tenor structure, and rate inputs with the target cluster's
market data. The bootstrap functions are commodity-agnostic.
