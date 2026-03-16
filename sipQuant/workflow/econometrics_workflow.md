# sipQuant econometrics — SIP Global Workflow Guide

## Purpose

The `econometrics` module provides regression, stationarity testing, structural break
detection, and Granger causality for commodity index methodology and proxy construction.

---

## Function Reference

| Function | Use Case |
|---|---|
| `ols` | Proxy regression, grade premium modelling |
| `huberRegression` | Outlier-robust regression for thin markets |
| `theilSen` | Non-parametric slope for sparse trade data |
| `madVol` | Robust rolling volatility (no outlier contamination) |
| `adfTest` / `kpssTest` | Stationarity screening before index construction |
| `chowTest` | Detect single structural break in index series |
| `baiPerron` | Multiple structural breaks, methodology review trigger |
| `cusum` | Ongoing parameter stability monitoring |
| `grangerCausality` | Test whether proxy leads/lags target market |
| `ljungBox` | Residual autocorrelation check after regression |

---

## Step-by-Step: Proxy Regression Workflow

### 1. Stationarity check before any regression

```python
import sipQuant as sq
import numpy as np

# Alberta Hay Premium weekly log returns
log_ret = np.diff(np.log(prices))

adf = sq.econometrics.adfTest(log_ret, regression='c')
kpss = sq.econometrics.kpssTest(log_ret, regression='c')
print(f"ADF: {adf['conclusion']}")
print(f"KPSS: {kpss['conclusion']}")
# Both should confirm stationarity of returns before regression
```

### 2. OLS proxy regression (grade premium vs. market factors)

```python
# Regress premium grade vs. standard grade price + volume
X = np.column_stack([standard_prices, volumes])
result = sq.econometrics.ols(premium_prices, X, addConst=True, robust=True, covType='HC3')
print(f"R²: {result['rSquared']:.3f}")
print(f"Coefficients: {result['coefficients']}")
print(f"P-values: {result['pValues']}")
```

### 3. Huber regression when outlier trades are present

```python
# In thin markets, occasional distressed trades contaminate OLS
# Use Huber regression for outlier-robust estimates
huber = sq.econometrics.huberRegression(premium_prices, X, addConst=True, delta=1.345)
print(f"Huber coefs: {huber['coefficients']}")

# Compare: if Huber and OLS coefficients diverge significantly,
# flag the series for data quality review
coef_diff = np.abs(huber['coefficients'] - result['coefficients'])
if coef_diff.max() > 0.1:
    print("WARNING: outlier influence detected — use Huber estimates")
```

### 4. Structural break detection (triggers methodology review)

```python
# Annual review: test for structural breaks in index series
chow = sq.econometrics.chowTest(index_values, market_factors, breakPoint=len(index_values)//2)
print(f"Chow F-stat: {chow['fStat']:.3f}, p-value: {chow['pValue']:.4f}")

# For full break search (multiple breaks)
bp = sq.econometrics.baiPerron(index_values, market_factors, maxBreaks=3)
print(f"Detected {bp['nBreaks']} structural breaks at indices: {bp['breakIndices']}")
```

### 5. CUSUM monitoring (ongoing stability)

```python
# Run monthly — flag if CUSUM exceeds critical bands
cusum = sq.econometrics.cusum(index_values, market_factors)
breaches = np.any(cusum['cusumStat'] > cusum['critBands']['upper']) or \
           np.any(cusum['cusumStat'] < cusum['critBands']['lower'])
if breaches:
    print("ALERT: CUSUM bands breached — parameter instability detected")
```

### 6. Granger causality for proxy validation

```python
# Does the broker quote Granger-cause the physical trade index?
gc = sq.econometrics.grangerCausality(index_values, broker_quotes, maxLag=4)
print(f"Granger test: {gc['conclusion']}")
# If yes: broker quotes are a valid leading proxy for the index
```

### 7. Residual diagnostics

```python
resid = result['residuals']
dw = sq.econometrics.durbinWatson(resid)
lb = sq.econometrics.ljungBox(resid, lags=10)
print(f"Durbin-Watson: {dw:.3f}")
print(f"Ljung-Box p-values: {[f'{p:.3f}' for p in lb['pValues']]}")
```

---

## Adapting to Any Cluster

Replace `premium_prices`, `standard_prices`, `volumes` with the target cluster's
price and volume series. The regression and test functions are market-agnostic.
For index methodology review, run steps 4–5 on a quarterly basis for every
active SIP cluster index.
