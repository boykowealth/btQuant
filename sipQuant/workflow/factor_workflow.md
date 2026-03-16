# sipQuant factor — SIP Global Workflow Guide

## Purpose

The `factor` module provides CAPM regression, rolling beta, and PCA-based statistical
factor extraction. Used for return attribution, proxy construction, and cross-cluster
correlation analysis.

---

## Step-by-Step: Factor Analysis for Commodity Clusters

### 1. CAPM regression: cluster vs. benchmark

```python
import sipQuant as sq
import numpy as np

# Asset: Alberta Hay Premium weekly returns
# Market: SIP Commodity Composite Index returns
asset_ret = np.diff(np.log(hay_prices))
market_ret = np.diff(np.log(composite_index))

result = sq.factor.capm(asset_ret, market_ret, rf=0.0)
print(f"Beta: {result['beta']:.3f}")
print(f"Alpha: {result['alpha']:.4f}")
print(f"R²: {result['rSquared']:.3f}")
```

### 2. Rolling beta for regime monitoring

```python
rolling = sq.factor.rollingBeta(asset_ret, market_ret, window=52, rf=0.0)
# rolling['betas']: 52-week rolling beta over time
# Use to detect when cluster correlation with composite changes (e.g., drought years)
```

### 3. PCA factors for multi-cluster book

```python
# Returns matrix: (T x n) where n = number of clusters
returns_matrix = np.column_stack([
    hay_ret, wheat_ret, canola_ret, barley_ret, oat_ret
])

factors = sq.factor.pcaFactors(returns_matrix, nFactors=3)
# factors['factors']: (T x 3) — 3 latent commodity factors
# factors['loadings']: (5 x 3) — how each cluster loads on each factor
# factors['explainedVariance']: variance explained by each factor

print(f"Factor 1 explains: {factors['explainedVariance'][0]*100:.1f}%")
```

### 4. Factor-based P&L decomposition

```python
# Position in Alberta Hay + Western Canadian Wheat
positions = np.array([500.0, -300.0])  # notional units

# Factor exposures = loadings^T @ positions
factor_exposures = factors['loadings'].T @ positions

# Factor P&L attribution
factor_returns_today = factors['factors'][-1]  # last day
pnl_by_factor = factor_exposures * factor_returns_today
```

---

## Adapting to Any Cluster

Build the returns matrix from any set of SIP cluster price series.
PCA factors automatically capture the dominant co-movement structure
across the clusters in that specific analysis window.
