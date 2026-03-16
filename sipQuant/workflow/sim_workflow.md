# sipQuant sim — SIP Global Workflow Guide

## Purpose

The `sim` module provides vectorized price process simulation for commodity markets.
Used for Monte Carlo pricing of OTC instruments, scenario generation, VaR estimation,
and stress testing across all SIP commodity clusters.

---

## Key Processes by Use Case

| Process | Use Case |
|---|---|
| `gbm` | Equity-like commodity prices (liquid futures) |
| `ou` | Mean-reverting commodities (hay, straw, seasonal crops) |
| `levyOu` | OU with spikes (electricity, feed grains with supply shocks) |
| `garch` | Volatility clustering in actively traded markets |
| `heston` | Stochastic vol for options on thin commodities |
| `arma` | Index construction with autocorrelated inputs |
| `markovSwitching` | Regime-switching (drought/normal, crop year transitions) |

---

## Step-by-Step: Monte Carlo OTC Pricing Workflow

### 1. Calibrate process to historical data

```python
import sipQuant as sq
import numpy as np

# Load price series for Alberta Hay Premium (or any cluster)
ps = sq.schema.PriceSeries(dates, prices, 'broker_a', 'alberta_hay_premium')

# Fit OU process (common for physical commodities)
fitResult = sq.fit.fitOU(ps['values'], dt=1/52)  # weekly data
print(fitResult)  # {'theta': ..., 'mu': ..., 'sigma': ...}
```

### 2. Simulate forward paths

```python
paths = sq.sim.ou(
    theta=fitResult['theta'],
    mu=fitResult['mu'],
    sigma=fitResult['sigma'],
    nSteps=52,       # 1 year weekly
    nSims=5000,
    x0=ps['values'][-1],   # start from current price
    dt=1/52
)
# paths: (5000 x 52) array
```

### 3. Price Asian swap on simulated paths

```python
# Average price over the year (Asian swap floating leg)
avgPrices = paths.mean(axis=1)  # arithmetic average per path
payoffs = np.maximum(avgPrices - fixedStrike, 0)
npv = np.exp(-r * T) * payoffs.mean()
```

### 4. GARCH volatility for risk management

```python
# Estimate current vol regime from recent returns
returns = np.diff(np.log(ps['values']))
garchFit = sq.fit.fitGarch(returns)

# Simulate vol-adjusted scenarios
scenarios = sq.sim.garch(
    omega=garchFit['omega'],
    alpha1=garchFit['alpha'],
    beta1=garchFit['beta'],
    nSteps=21,   # 1 month
    nSims=10000
)
```

### 5. Heston for stochastic vol options

```python
hestonFit = sq.fit.fitHeston(ps['values'], dt=1/52)
simResult = sq.sim.heston(
    mu=hestonFit['mu'],
    kappa=hestonFit['kappa'],
    theta=hestonFit['theta'],
    sigma=hestonFit['sigma'],
    rho=hestonFit['rho'],
    s0=ps['values'][-1],
    v0=hestonFit['v0'],
    nSteps=52,
    nSims=5000
)
pricePaths = simResult['prices']  # (5000 x 52)
```

---

## Adapting to Any Cluster

Replace process choice and calibration data with the target cluster:
- **Hay/straw**: OU (strong mean reversion, seasonal mu)
- **Feed grains**: Lévy-OU (jump risk at crop reports)
- **Canola/oilseeds**: GARCH (vol clustering)
- **Pulse crops**: Markov switching (on/off crop year regimes)
- **Carbon credits**: GBM (no natural mean reversion)
