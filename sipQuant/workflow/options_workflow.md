# sipQuant options — SIP Global Workflow Guide

## Purpose

The `options` module prices European and exotic options on physical commodity underlyings.
Used for OTC option structuring, vol surface construction, and delta hedging across all
SIP commodity clusters.

---

## Function Reference

| Function | Use Case |
|---|---|
| `blackScholes` | European call/put on any cluster's forward price |
| `spread` | Grade spread options (premium vs. standard hay) |
| `barrier` | Knock-out options on commodity forwards |
| `asian` | Monthly average settlement (most physical OTC deals) |
| `binomial` | American early-exercise pricing |
| `impliedVol` | Backing out vol from broker quotes |
| `monteCarlo` | Exotic payoff pricing |

---

## Step-by-Step: OTC Option Workflow

### 1. Price a European cap on Alberta Hay

```python
import sipQuant as sq
import numpy as np

# Forward price from curve
F = 192.0   # forward at delivery tenor
K = 195.0   # cap strike
T = 0.5     # 6-month tenor
r = 0.046
sigma = 0.18  # vol from vol surface

cap = sq.options.blackScholes(S=F, K=K, T=T, r=r, sigma=sigma, optType='call')
print(f"Cap premium: {cap['price']:.4f} $/unit")
print(f"Delta: {cap['delta']:.4f}")
```

### 2. Build implied vol surface from broker quotes

```python
# Broker quotes at various strikes and tenors
market_prices = np.array([...])  # observed premiums
strikes = np.array([170, 180, 190, 200, 210])
tenors  = np.array([0.25, 0.5, 1.0])

vol_surface = np.zeros((len(tenors), len(strikes)))
for i, T in enumerate(tenors):
    for j, K in enumerate(strikes):
        vol_surface[i, j] = sq.options.impliedVol(
            price=market_prices[i, j], S=F, K=K, T=T, r=r
        )

surface = sq.bootstrap.volSurface(strikes, tenors, vol_surface)
```

### 3. Price monthly-average Asian option (common in physical OTC)

```python
# Most physical commodity OTC deals settle on monthly average price
asian = sq.options.asian(
    S=187.5, K=190.0, T=1/12,
    r=0.046, sigma=0.18,
    nSteps=20,    # 20 trading days in month
    nSims=10000,
    optType='call', avgType='arithmetic'
)
print(f"Asian call: {asian['price']:.4f}, StdErr: {asian['stderr']:.4f}")
```

### 4. Barrier option for knock-out floor

```python
# Down-and-out put: protection that knocks out if price falls below barrier
barrier_put = sq.options.barrier(
    S=187.5, K=185.0, T=0.25, r=0.046, sigma=0.18,
    barrierLevel=170.0, optType='put', barrierType='down-and-out'
)
print(f"Barrier put: {barrier_put['price']:.4f}")
print(f"Delta: {barrier_put['delta']:.4f}, Vega: {barrier_put['vega']:.4f}")
```

### 5. Delta hedge sizing

```python
# Scale option delta to hedge units
option_notional = 5000.0  # units (tonnes / bales)
hedge_units = -cap['delta'] * option_notional
print(f"Hedge {hedge_units:.0f} units in the forward market")
```

---

## Adapting to Any Cluster

Replace `S` (spot/forward), `K` (strike), `sigma` (cluster vol), and `r` with
the target cluster's parameters. The option pricing models are commodity-agnostic.
Vol surfaces and implied vol calibration work identically for any SIP cluster.
