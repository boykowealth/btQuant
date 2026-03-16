# sipQuant portfolio — SIP Global Workflow Guide

## Purpose

The `portfolio` module optimises allocation across multiple commodity clusters
and OTC positions. Used for strategic allocation decisions, risk budget management,
and Black-Litterman overlays when SIP has directional views.

---

## Function Reference

| Function | Use Case |
|---|---|
| `meanVariance` | Classic Markowitz minimum-variance or target-return |
| `efficientFrontier` | Plot risk/return trade-off across cluster universe |
| `hrp` | Hierarchical Risk Parity — robust to estimation error |
| `riskParity` | Equal risk contribution (Spinu/log-barrier CCD) |
| `blackLitterman` | Incorporate SIP analyst views into allocation |
| `maxSharpe` | Highest risk-adjusted return portfolio |
| `minCvar` | Minimum CVaR — tail-risk-aware allocation |

---

## Step-by-Step: Multi-Cluster Allocation Workflow

### 1. Compute returns and covariance

```python
import sipQuant as sq
import numpy as np

# Weekly log returns for 5 clusters
# rows = weeks, cols = clusters (hay, wheat, canola, barley, oat)
returns = np.column_stack([
    np.diff(np.log(hay_prices)),
    np.diff(np.log(wheat_prices)),
    np.diff(np.log(canola_prices)),
    np.diff(np.log(barley_prices)),
    np.diff(np.log(oat_prices)),
])  # shape (T, 5)

mu  = returns.mean(axis=0) * 52     # annualised
cov = np.cov(returns.T) * 52        # annualised
```

### 2. Minimum variance portfolio (long-only)

```python
mv = sq.portfolio.meanVariance(mu, cov, allowShort=False)
print(f"Weights:    {np.round(mv['weights'], 3)}")
print(f"Return:     {mv['return']:.4f}")
print(f"Volatility: {mv['volatility']:.4f}")
print(f"Sharpe:     {mv['sharpe']:.3f}")
```

### 3. HRP — robust to noisy covariance estimates

```python
# HRP is preferred when data is limited (< 3 years)
hrp = sq.portfolio.hrp(returns)
print(f"HRP weights: {np.round(hrp['weights'], 3)}")
print(f"Asset order: {hrp['order']}")
```

### 4. Risk Parity — equal risk contribution

```python
rp = sq.portfolio.riskParity(cov)
print(f"RP weights: {np.round(rp['weights'], 3)}")
print(f"Risk contribs: {np.round(rp['riskContributions'], 4)}")
# All risk contributions should be approximately equal
```

### 5. Black-Litterman overlay with SIP analyst views

```python
# View 1: Alberta Hay outperforms Western Canadian Wheat by 3% annualised
# P row: long hay (col 0), short wheat (col 1)
P = np.array([[1.0, -1.0, 0.0, 0.0, 0.0]])
Q = np.array([0.03])  # 3% annualised outperformance

bl = sq.portfolio.blackLitterman(mu, cov, P, Q, tau=0.05)
print(f"BL posterior mu: {np.round(bl['muBL'], 4)}")
print(f"BL weights:      {np.round(bl['weights'], 3)}")
```

### 6. Efficient frontier for risk budget discussion

```python
frontier = sq.portfolio.efficientFrontier(mu, cov, nPoints=30, allowShort=False)

# Display risk/return trade-off
for r, v in zip(frontier['returns'], frontier['volatilities']):
    print(f"  Return: {r:.4f}  Vol: {v:.4f}  Sharpe: {r/v:.3f}")
```

### 7. MinCVaR for tail-risk-sensitive mandates

```python
# Use when client mandate specifies CVaR budget
mc = sq.portfolio.minCvar(returns, alpha=0.05)
print(f"MinCVaR weights: {np.round(mc['weights'], 3)}")
print(f"Portfolio CVaR:  {mc['cvar']:.4f}")
```

---

## Adapting to Any Cluster Universe

Build `returns` from any set of SIP cluster `PriceSeries` objects. The portfolio
functions are agnostic to commodity type. For single-cluster mandates, pass a
1-asset returns matrix — functions handle the degenerate case gracefully.
