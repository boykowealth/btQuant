# sipQuant ml — SIP Global Workflow Guide

## Purpose

The `ml` module provides tree-based regression and clustering for commodity price
prediction, segment identification, and sparse market data imputation.

---

## Step-by-Step: ML in the SIP Index Pipeline

### 1. Regression tree for grade premium prediction

```python
import sipQuant as sq
import numpy as np

# Features: moisture, dockage, test weight, days since harvest
X_train = np.column_stack([moisture, dockage, testWeight, daysHarvest])
y_train = grade_premiums  # observed premium over base

tree = sq.ml.regressionTree(X_train, y_train, maxDepth=4, minSamplesSplit=5)
# Predict for new observations
y_pred = tree['predict'](X_new)
```

### 2. Random forest for robust proxy regression

```python
# When only sparse trades are available, RF can impute missing grades
forest = sq.ml.randomForest(
    X_train, y_train,
    nTrees=50,
    maxDepth=4,
    maxFeatures=2,   # sqrt(nFeatures)
    seed=42
)

importances = forest['featureImportances']
print("Feature importances:", importances)
# Use most important features to focus data collection effort
```

### 3. Gradient boosting for index value prediction

```python
# Predict index value from proxy observable features
# when direct trades are not available on a given day
gb = sq.ml.gradientBoosting(
    X_train, y_train,
    nEstimators=50,
    learningRate=0.1,
    maxDepth=3
)
proxy_index_value = gb['predict'](X_today)
```

### 4. k-Means for market segmentation

```python
# Cluster commodity markets by price behaviour
# Features: volatility, mean reversion speed, liquidity score
market_features = np.column_stack([vol_array, theta_array, liquidity_score])

clusters = sq.ml.kMeans(market_features, k=4, seed=42)
# clusters['labels']: which cluster each market belongs to
# Use to group similar markets for shared index methodology
print(f"Cluster centroids:\n{clusters['centroids']}")
```

---

## Adapting to Any Cluster

Feed cluster-specific trade data and grade attributes as `X_train`.
The tree/forest/boosting models learn cluster-specific pricing structure
without any hardcoded assumptions about commodity type.
