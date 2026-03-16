"""
sipQuant.ml — Machine learning models for quantitative finance.
SIP Global (Systematic Index Partners)

Pure NumPy. No external dependencies.

Functions
---------
regressionTree    : CART regression tree.
randomForest      : Random forest of regression trees.
gradientBoosting  : Gradient boosting (MSE, residuals as gradient).
kMeans            : k-Means clustering with k-means++ initialisation.
"""

import numpy as np


# ---------------------------------------------------------------------------
# CART tree builder (private)
# ---------------------------------------------------------------------------

def _buildTree(X, y, maxDepth, minSamplesSplit, depth=0):
    """Recursive CART regression tree builder.

    Parameters
    ----------
    X               : (n, p) float array.
    y               : (n,) float array.
    maxDepth        : int — maximum tree depth.
    minSamplesSplit : int — minimum samples required to attempt a split.
    depth           : int — current recursion depth.

    Returns
    -------
    Nested dict with either:
        leaf node  — {'leaf': True, 'value': float}
        split node — {'leaf': False, 'feature': int, 'threshold': float,
                       'left': ..., 'right': ..., 'nSamples': int,
                       'impurityDecrease': float}
    """
    n, p = X.shape

    # Leaf conditions
    if depth >= maxDepth or n < minSamplesSplit or n == 1:
        return {'leaf': True, 'value': float(np.mean(y))}

    bestMse = np.inf
    bestFeature = None
    bestThreshold = None
    nodeMse = float(np.var(y) * n)

    for feature in range(p):
        xCol = X[:, feature]
        thresholds = np.unique(xCol)
        if len(thresholds) < 2:
            continue

        # Mid-points between unique values
        mids = (thresholds[:-1] + thresholds[1:]) / 2.0

        for thresh in mids:
            leftMask = xCol <= thresh
            rightMask = ~leftMask
            nLeft = int(np.sum(leftMask))
            nRight = n - nLeft

            if nLeft == 0 or nRight == 0:
                continue

            yLeft = y[leftMask]
            yRight = y[rightMask]
            mse = (nLeft * float(np.var(yLeft)) + nRight * float(np.var(yRight)))

            if mse < bestMse:
                bestMse = mse
                bestFeature = feature
                bestThreshold = thresh

    if bestFeature is None:
        return {'leaf': True, 'value': float(np.mean(y))}

    leftMask = X[:, bestFeature] <= bestThreshold
    rightMask = ~leftMask

    leftChild = _buildTree(X[leftMask], y[leftMask], maxDepth, minSamplesSplit, depth + 1)
    rightChild = _buildTree(X[rightMask], y[rightMask], maxDepth, minSamplesSplit, depth + 1)

    return {
        'leaf': False,
        'feature': bestFeature,
        'threshold': bestThreshold,
        'left': leftChild,
        'right': rightChild,
        'nSamples': n,
        'impurityDecrease': float((nodeMse - bestMse) / n),
    }


def _predictTree(tree, x):
    """Predict a single sample by traversing a tree dict.

    Parameters
    ----------
    tree : dict — tree returned by _buildTree.
    x    : (p,) array — single input sample.

    Returns
    -------
    float — predicted value.
    """
    while not tree['leaf']:
        if x[tree['feature']] <= tree['threshold']:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree['value']


def _featureImportances(tree, nFeatures):
    """Accumulate feature importances from a tree by traversal."""
    importances = np.zeros(nFeatures)

    def _traverse(node):
        if node['leaf']:
            return
        importances[node['feature']] += node.get('impurityDecrease', 0.0) * node.get('nSamples', 1)
        _traverse(node['left'])
        _traverse(node['right'])

    _traverse(tree)
    total = np.sum(importances)
    if total > 0:
        importances /= total
    return importances


# ---------------------------------------------------------------------------
# Regression tree
# ---------------------------------------------------------------------------

def regressionTree(X, y, maxDepth=5, minSamplesSplit=5):
    """CART regression tree.

    Parameters
    ----------
    X               : array-like, shape (n, p).
    y               : array-like, shape (n,).
    maxDepth        : int — maximum depth of the tree.
    minSamplesSplit : int — minimum samples to consider a split.

    Returns
    -------
    dict with keys:
        tree    — nested dict representing the fitted tree.
        predict — callable(X_new) -> (n,) predictions.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    tree = _buildTree(X, y, maxDepth, minSamplesSplit)

    def predict(Xnew):
        Xnew = np.asarray(Xnew, dtype=float)
        if Xnew.ndim == 1:
            return np.array([_predictTree(tree, Xnew)])
        return np.array([_predictTree(tree, xi) for xi in Xnew])

    return {
        'tree': tree,
        'predict': predict,
    }


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------

def randomForest(X, y, nTrees=50, maxDepth=4, minSamplesSplit=5, maxFeatures=None, seed=None):
    """Random forest of regression trees with bootstrap sampling.

    Parameters
    ----------
    X               : array-like, shape (n, p).
    y               : array-like, shape (n,).
    nTrees          : int — number of trees in the forest.
    maxDepth        : int — maximum depth per tree.
    minSamplesSplit : int — minimum samples per split.
    maxFeatures     : int or None — features per split candidate set (default: p // 3 or 1).
    seed            : int or None — RNG seed.

    Returns
    -------
    dict with keys:
        trees              — list of fitted tree dicts.
        featureImportances — (p,) mean normalised importance across trees.
        predict            — callable(X_new) -> (n,) predictions.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape

    if maxFeatures is None:
        maxFeatures = max(1, p // 3)
    maxFeatures = int(maxFeatures)

    rng = np.random.default_rng(seed)
    trees = []
    importancesAll = np.zeros((nTrees, p))

    for t in range(nTrees):
        # Bootstrap sample
        idx = rng.integers(0, n, size=n)
        Xboot = X[idx]
        yboot = y[idx]

        # Random feature subspace: build wrapper that restricts features
        featIdx = rng.choice(p, size=min(maxFeatures, p), replace=False)
        Xsub = Xboot[:, featIdx]

        treeDict = _buildTree(Xsub, yboot, maxDepth, minSamplesSplit)

        # Store feature index mapping for prediction
        trees.append({'tree': treeDict, 'featIdx': featIdx})

        imp = _featureImportances(treeDict, len(featIdx))
        for j, fi in enumerate(featIdx):
            importancesAll[t, fi] += imp[j]

    featureImportances = np.mean(importancesAll, axis=0)
    total = np.sum(featureImportances)
    if total > 0:
        featureImportances /= total

    def predict(Xnew):
        Xnew = np.asarray(Xnew, dtype=float)
        if Xnew.ndim == 1:
            Xnew = Xnew.reshape(1, -1)
        preds = np.zeros((len(trees), len(Xnew)))
        for t, td in enumerate(trees):
            Xsub = Xnew[:, td['featIdx']]
            preds[t] = np.array([_predictTree(td['tree'], xi) for xi in Xsub])
        return np.mean(preds, axis=0)

    return {
        'trees': trees,
        'featureImportances': featureImportances,
        'predict': predict,
    }


# ---------------------------------------------------------------------------
# Gradient boosting
# ---------------------------------------------------------------------------

def gradientBoosting(X, y, nEstimators=50, learningRate=0.1, maxDepth=3, minSamplesSplit=5):
    """Gradient boosting with MSE loss (negative gradient = residuals).

    Fits sequentially: F_m(x) = F_{m-1}(x) + lr * h_m(x)
    where h_m is a regression tree fitted on the residuals.

    Parameters
    ----------
    X               : array-like, shape (n, p).
    y               : array-like, shape (n,).
    nEstimators     : int — number of boosting rounds.
    learningRate    : float — shrinkage factor.
    maxDepth        : int — maximum tree depth.
    minSamplesSplit : int — minimum samples per split.

    Returns
    -------
    dict with keys:
        estimators        — list of tree dicts.
        predict           — callable(X_new) -> (n,) predictions.
        trainingResiduals — (n,) final residuals after all estimators.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)

    # Initialise with mean
    F = np.full(n, np.mean(y))
    estimators = []

    for _ in range(nEstimators):
        residuals = y - F
        treeResult = regressionTree(X, residuals, maxDepth=maxDepth, minSamplesSplit=minSamplesSplit)
        hm = treeResult['predict'](X)
        F = F + learningRate * hm
        estimators.append(treeResult['tree'])

    initValue = float(np.mean(y))

    def predict(Xnew):
        Xnew = np.asarray(Xnew, dtype=float)
        if Xnew.ndim == 1:
            Xnew = Xnew.reshape(1, -1)
        nNew = len(Xnew)
        Fpred = np.full(nNew, initValue)
        for tree in estimators:
            hm = np.array([_predictTree(tree, xi) for xi in Xnew])
            Fpred += learningRate * hm
        return Fpred

    trainingResiduals = y - F

    return {
        'estimators': estimators,
        'predict': predict,
        'trainingResiduals': trainingResiduals,
    }


# ---------------------------------------------------------------------------
# k-Means
# ---------------------------------------------------------------------------

def kMeans(X, k, maxIter=300, tol=1e-4, seed=None):
    """k-Means clustering with k-means++ initialisation.

    Parameters
    ----------
    X       : array-like, shape (n, p).
    k       : int — number of clusters.
    maxIter : int — maximum EM iterations.
    tol     : float — centroid shift tolerance for convergence.
    seed    : int or None — RNG seed.

    Returns
    -------
    dict with keys:
        labels    — (n,) int array of cluster assignments.
        centroids — (k, p) centroid array.
        inertia   — float, sum of squared distances to assigned centroids.
        nIter     — int, number of iterations until convergence.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    k = int(k)

    rng = np.random.default_rng(seed)

    # k-means++ initialisation
    centroids = np.empty((k, p))
    firstIdx = rng.integers(0, n)
    centroids[0] = X[firstIdx]

    for c in range(1, k):
        diffs = X[:, None, :] - centroids[None, :c, :]  # (n, c, p)
        distSq = np.min(np.sum(diffs ** 2, axis=2), axis=1)  # (n,)
        probs = distSq / np.sum(distSq)
        chosenIdx = rng.choice(n, p=probs)
        centroids[c] = X[chosenIdx]

    labels = np.zeros(n, dtype=int)

    for iteration in range(maxIter):
        # Assignment step
        diffs = X[:, None, :] - centroids[None, :, :]  # (n, k, p)
        distSq = np.sum(diffs ** 2, axis=2)             # (n, k)
        newLabels = np.argmin(distSq, axis=1)

        # Update step
        newCentroids = np.empty_like(centroids)
        for c in range(k):
            mask = newLabels == c
            if np.sum(mask) > 0:
                newCentroids[c] = np.mean(X[mask], axis=0)
            else:
                newCentroids[c] = centroids[c]

        shift = float(np.max(np.linalg.norm(newCentroids - centroids, axis=1)))
        labels = newLabels
        centroids = newCentroids

        if shift < tol:
            break

    # Compute inertia
    assignedCentroids = centroids[labels]
    inertia = float(np.sum(np.sum((X - assignedCentroids) ** 2, axis=1)))

    return {
        'labels': labels,
        'centroids': centroids,
        'inertia': inertia,
        'nIter': iteration + 1,
    }
