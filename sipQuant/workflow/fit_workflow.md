# sipQuant fit — SIP Global Workflow Guide

## Purpose

The `fit` module calibrates stochastic process and distribution models to observed
commodity price data. Results feed directly into sim, otc, and risk modules.

---

## Calibration Map

| Function | Input | Output | Downstream Use |
|---|---|---|---|
| `fitOU` | Price series | theta, mu, sigma | `sim.ou`, mean reversion pricing |
| `fitGarch` | Returns series | omega, alpha, beta | `sim.garch`, VaR estimation |
| `fitHeston` | Price series | kappa, theta, sigma, rho | `sim.heston`, options pricing |
| `fitCopula` | Uniform marginals | copula params | Joint distribution, CVaR |
| `fitDistribution` | Any series | dist params | Marginal distribution analysis |

---

## Step-by-Step: Full Calibration Pipeline

### 1. Fit OU to physical commodity prices

```python
import sipQuant as sq
import numpy as np

# Alberta Hay Premium — weekly closing prices
ps = sq.schema.PriceSeries(dates, prices, 'broker_a', 'alberta_hay_premium')
returns = np.diff(np.log(ps['values']))

fit = sq.fit.fitOU(ps['values'], dt=1/52)
# Result: {'theta': 2.4, 'mu': 185.0, 'sigma': 8.2, 'logLik': -124.3}

# Half-life of mean reversion:
halfLife = np.log(2) / fit['theta']  # in weeks
print(f"Mean reversion half-life: {halfLife:.1f} weeks")
```

### 2. Fit GARCH to returns for vol modeling

```python
garchFit = sq.fit.fitGarch(returns)
# Result: {'omega': 0.0001, 'alpha': 0.08, 'beta': 0.88, 'logLik': 342.1, 'aic': -680.2}

persistence = garchFit['alpha'] + garchFit['beta']
print(f"GARCH persistence: {persistence:.3f}")
```

### 3. Multivariate copula for joint commodity risk

```python
# Two clusters: Alberta Hay + Western Canadian Feed Wheat
hay_returns = np.diff(np.log(hay_prices))
wheat_returns = np.diff(np.log(wheat_prices))

# Convert to uniform marginals via rank transform
n = len(hay_returns)
u_hay = (np.argsort(np.argsort(hay_returns)) + 1) / (n + 1)
u_wheat = (np.argsort(np.argsort(wheat_returns)) + 1) / (n + 1)

data = np.column_stack([u_hay, u_wheat])

# Fit t-copula (captures tail dependence in commodity pairs)
copulaFit = sq.fit.fitCopula(data, copulaType='t')
print(f"t-copula df: {copulaFit['param']['df']:.1f}")
print(f"Lower tail dependence: {copulaFit['tailDep']['lower']:.3f}")
```

### 4. Tail distribution for risk capital

```python
# Fit heavy-tailed distribution to daily P&L
distFit = sq.fit.fitDistribution(daily_pnl, distType='t')
# Result: {'params': {'mu': ..., 'sigma': ..., 'df': ...}, 'aic': ..., 'bic': ...}
```

### 5. Model selection via AIC/BIC

```python
models = ['normal', 'lognormal', 't', 'gamma']
fits = {m: sq.fit.fitDistribution(returns, distType=m) for m in models}
best = min(fits, key=lambda m: fits[m]['aic'])
print(f"Best model by AIC: {best}")
```

---

## Adapting to Any Cluster

Calibrate with the price series from the target cluster's `PriceSeries` schema object.
The fit functions are model-agnostic with respect to the underlying commodity.
