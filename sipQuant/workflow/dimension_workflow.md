# sipQuant dimension — SIP Global Workflow Guide

## Purpose

The `dimension` module provides dimensionality reduction for multi-cluster price
data exploration, risk factor compression, and visualization of commodity market
structure.

---

## Step-by-Step: Dimensionality Reduction for Commodity Analysis

### 1. PCA on multi-cluster returns

```python
import sipQuant as sq
import numpy as np

# Returns matrix: (T x n_clusters)
returns = np.column_stack([hay_ret, wheat_ret, canola_ret, barley_ret, oat_ret,
                            silage_ret, straw_ret, pulse_ret])

result = sq.dimension.pca(returns, nComponents=3)

print("Explained variance ratio:", result['explainedVarianceRatio'])
print("Total explained:", result['explainedVarianceRatio'][:3].sum())

# Component 1 typically = "broad commodity" factor
# Component 2 typically = "crop vs. forage" split
# Component 3 typically = "supply shock" residual
```

### 2. Kernel PCA for nonlinear structure

```python
# When linear PCA misses nonlinear co-movement
kpca_result = sq.dimension.kernelPca(returns, nComponents=2, kernel='rbf', gamma=0.5)
embedding = kpca_result['scores']  # (T x 2) nonlinear embedding
```

### 3. ICA for independent source separation

```python
# Separate independent price drivers (weather, logistics, demand)
ica_result = sq.dimension.ica(returns, nComponents=3)
independent_sources = ica_result['sources']  # (T x 3)
```

### 4. t-SNE for exploratory visualization

```python
# Visualize market observations in 2D (for analyst review)
tsne_result = sq.dimension.tsne(returns, nComponents=2, perplexity=30, nIter=500, seed=42)
embedding_2d = tsne_result['embedding']  # (T x 2)

# Plot: color by crop year, drought year, etc.
# Reveals regime structure not visible in raw correlations
```

---

## Adapting to Any Cluster

Build the returns matrix from any combination of SIP cluster price series.
PCA and ICA are linear; kernel PCA and t-SNE capture nonlinear structure.
For index methodology review, PCA components help explain why certain
clusters move together and justify constituent groupings.
