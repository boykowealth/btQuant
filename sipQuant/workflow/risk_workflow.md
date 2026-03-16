# sipQuant risk — SIP Global Workflow Guide

## Purpose

The `risk` module computes VaR, CVaR, drawdown, and performance ratios for commodity
trading books and index portfolios. Used for daily risk reporting, margin estimation,
and IOSCO-aligned risk disclosure.

---

## Function Reference

| Function | Use Case |
|---|---|
| `var` | Daily/weekly Value at Risk on book or cluster |
| `cvar` | Expected Shortfall (CVaR) — regulatory risk measure |
| `maxDrawdown` | Drawdown analysis for index and strategy |
| `sortino` | Downside-risk-adjusted return |
| `calmar` | Return-to-drawdown ratio for fund reporting |
| `hillEstimator` | Tail index — measures heaviness of loss distribution |
| `portfolioVar` | VaR across multi-cluster portfolio |
| `rollingVol` | Rolling annualised volatility |
| `beta` | CAPM beta of cluster vs. composite |
| `trackingError` | Index replication quality |

---

## Step-by-Step: Daily Risk Reporting Workflow

### 1. Historical VaR on the physical book

```python
import sipQuant as sq
import numpy as np

# Daily P&L series from book (use pnlAttribution output)
daily_pnl = np.array([...])   # positive = profit, negative = loss

var_result = sq.risk.var(daily_pnl, alpha=0.05, method='historical')
print(f"1-day 95% VaR:  {var_result['var']:.2f}")

cvar_result = sq.risk.cvar(daily_pnl, alpha=0.05, method='historical')
print(f"1-day 95% CVaR: {cvar_result['cvar']:.2f}")
```

### 2. Portfolio VaR across multi-cluster positions

```python
# Returns matrix: (T x n_clusters), weights from portfolio module
returns_matrix = np.column_stack([hay_ret, wheat_ret, canola_ret])
weights = sq.portfolio.hrp(returns_matrix)['weights']

pvar = sq.risk.portfolioVar(weights, returns_matrix, alpha=0.05)
print(f"Portfolio VaR:  {pvar['var']:.4f}")
print(f"Portfolio CVaR: {pvar['cvar']:.4f}")
```

### 3. Drawdown analysis on index series

```python
# SIP-AHI index values
index_prices = np.array([...])

dd = sq.risk.maxDrawdown(index_prices)
print(f"Max drawdown:   {dd['maxDrawdown']:.4f} ({dd['maxDrawdown']*100:.1f}%)")
print(f"Peak at index:  {dd['peakIdx']}")
print(f"Trough at:      {dd['troughIdx']}")
```

### 4. Performance ratios for investor reporting

```python
index_returns = np.diff(np.log(index_prices))

sortino = sq.risk.sortino(index_returns, rf=0.0, periodsPerYear=52)
calmar  = sq.risk.calmar(index_returns, periodsPerYear=52)

print(f"Sortino ratio: {sortino['sortino']:.3f}")
print(f"Calmar ratio:  {calmar['calmar']:.3f}")
print(f"Annual return: {calmar['annualReturn']:.4f}")
```

### 5. Tail risk — Hill estimator for extreme events

```python
# Are commodity losses heavy-tailed?
losses = -daily_pnl[daily_pnl < 0]   # positive loss values
hill = sq.risk.hillEstimator(losses, threshold=None)  # auto: top 10%
print(f"Tail index (xi): {hill['xi']:.3f}")
# xi > 0.5: very heavy tail (power law); xi < 0.25: close to normal
```

### 6. Rolling volatility for vol regime monitoring

```python
weekly_returns = np.diff(np.log(index_prices))
rolling_vol = sq.risk.rollingVol(weekly_returns, window=13)  # 13-week rolling

# Alert if current vol exceeds 1.5x recent average
current_vol = rolling_vol[-1]
avg_vol = rolling_vol[-52:].mean()
if current_vol > 1.5 * avg_vol:
    print(f"ALERT: vol spike {current_vol:.3f} vs avg {avg_vol:.3f}")
```

### 7. Beta and tracking error for benchmark reporting

```python
composite_ret = np.diff(np.log(composite_index))
cluster_ret   = np.diff(np.log(cluster_prices))

b = sq.risk.beta(cluster_ret, composite_ret)
print(f"Beta vs composite: {b['beta']:.3f}")
print(f"Alpha (annualised): {b['alpha']*52:.4f}")

# Index replication quality
te = sq.risk.trackingError(index_ret, benchmark_ret)
print(f"Tracking error: {te['trackingError']:.4f}")
print(f"Info ratio:     {te['informationRatio']:.3f}")
```

---

## Adapting to Any Cluster

Replace `daily_pnl`, `returns_matrix`, and `index_prices` with the target cluster's
data from its `PriceSeries` schema object. The risk functions are entirely
commodity-agnostic and apply identically to any SIP cluster or composite index.
