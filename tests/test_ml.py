"""
Tests for sipQuant.ml
SIP Global (Systematic Index Partners)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
import sipQuant as sq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def regressionData():
    rng = np.random.default_rng(0)
    n = 100
    X = rng.standard_normal((n, 4))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + rng.normal(0, 0.5, n)
    return X, y


@pytest.fixture
def clusterData():
    rng = np.random.default_rng(1)
    # Three well-separated clusters
    c1 = rng.standard_normal((30, 2)) + np.array([5.0, 0.0])
    c2 = rng.standard_normal((30, 2)) + np.array([-5.0, 0.0])
    c3 = rng.standard_normal((30, 2)) + np.array([0.0, 5.0])
    X = np.vstack([c1, c2, c3])
    return X


# ---------------------------------------------------------------------------
# Regression tree
# ---------------------------------------------------------------------------

class TestRegressionTree:
    def test_predict_callable(self, regressionData):
        X, y = regressionData
        result = sq.ml.regressionTree(X, y)
        assert callable(result['predict'])

    def test_training_mse_below_total_variance(self, regressionData):
        X, y = regressionData
        result = sq.ml.regressionTree(X, y, maxDepth=5)
        preds = result['predict'](X)
        mse = float(np.mean((preds - y) ** 2))
        totalVar = float(np.var(y))
        assert mse < totalVar

    def test_predict_shape(self, regressionData):
        X, y = regressionData
        result = sq.ml.regressionTree(X, y)
        preds = result['predict'](X)
        assert preds.shape == (len(y),)

    def test_tree_key_exists(self, regressionData):
        X, y = regressionData
        result = sq.ml.regressionTree(X, y)
        assert 'tree' in result

    def test_predict_single_sample(self, regressionData):
        X, y = regressionData
        result = sq.ml.regressionTree(X, y)
        pred = result['predict'](X[0])
        assert np.isfinite(pred[0])

    def test_deep_tree_lower_mse_than_shallow(self, regressionData):
        X, y = regressionData
        r_shallow = sq.ml.regressionTree(X, y, maxDepth=1)
        r_deep = sq.ml.regressionTree(X, y, maxDepth=8)
        mse_shallow = float(np.mean((r_shallow['predict'](X) - y) ** 2))
        mse_deep = float(np.mean((r_deep['predict'](X) - y) ** 2))
        assert mse_deep <= mse_shallow

    def test_predict_finite_on_test_set(self, regressionData):
        X, y = regressionData
        rng = np.random.default_rng(99)
        Xtest = rng.standard_normal((20, X.shape[1]))
        result = sq.ml.regressionTree(X, y)
        preds = result['predict'](Xtest)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# Random forest
# ---------------------------------------------------------------------------

class TestRandomForest:
    def test_predict_callable(self, regressionData):
        X, y = regressionData
        result = sq.ml.randomForest(X, y, nTrees=10, seed=0)
        assert callable(result['predict'])

    def test_feature_importances_shape(self, regressionData):
        X, y = regressionData
        result = sq.ml.randomForest(X, y, nTrees=10, seed=0)
        assert result['featureImportances'].shape == (X.shape[1],)

    def test_feature_importances_sum_to_one(self, regressionData):
        X, y = regressionData
        result = sq.ml.randomForest(X, y, nTrees=10, seed=0)
        total = float(np.sum(result['featureImportances']))
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_predict_shape(self, regressionData):
        X, y = regressionData
        result = sq.ml.randomForest(X, y, nTrees=10, seed=0)
        preds = result['predict'](X)
        assert preds.shape == (len(y),)

    def test_trees_count(self, regressionData):
        X, y = regressionData
        result = sq.ml.randomForest(X, y, nTrees=15, seed=0)
        assert len(result['trees']) == 15

    def test_predict_finite(self, regressionData):
        X, y = regressionData
        result = sq.ml.randomForest(X, y, nTrees=10, seed=1)
        preds = result['predict'](X)
        assert np.all(np.isfinite(preds))

    def test_forest_outperforms_single_tree(self, regressionData):
        X, y = regressionData
        rng = np.random.default_rng(5)
        Xtest = rng.standard_normal((30, X.shape[1]))
        ytest = 3.0 * Xtest[:, 0] - 2.0 * Xtest[:, 1] + 0.5 * Xtest[:, 2]
        tree = sq.ml.regressionTree(X, y, maxDepth=3)
        forest = sq.ml.randomForest(X, y, nTrees=30, maxDepth=3, seed=0)
        mse_tree = float(np.mean((tree['predict'](Xtest) - ytest) ** 2))
        mse_forest = float(np.mean((forest['predict'](Xtest) - ytest) ** 2))
        # Forest should generally do better — allow small tolerance
        assert mse_forest <= mse_tree * 2.0  # loose bound


# ---------------------------------------------------------------------------
# Gradient boosting
# ---------------------------------------------------------------------------

class TestGradientBoosting:
    def test_predict_callable(self, regressionData):
        X, y = regressionData
        result = sq.ml.gradientBoosting(X, y, nEstimators=10)
        assert callable(result['predict'])

    def test_predict_shape(self, regressionData):
        X, y = regressionData
        result = sq.ml.gradientBoosting(X, y, nEstimators=10)
        preds = result['predict'](X)
        assert preds.shape == (len(y),)

    def test_training_residuals_shape(self, regressionData):
        X, y = regressionData
        result = sq.ml.gradientBoosting(X, y, nEstimators=10)
        assert result['trainingResiduals'].shape == (len(y),)

    def test_estimators_count(self, regressionData):
        X, y = regressionData
        result = sq.ml.gradientBoosting(X, y, nEstimators=20)
        assert len(result['estimators']) == 20

    def test_more_estimators_lower_training_mse(self, regressionData):
        X, y = regressionData
        r10 = sq.ml.gradientBoosting(X, y, nEstimators=5)
        r100 = sq.ml.gradientBoosting(X, y, nEstimators=50)
        mse10 = float(np.mean((r10['predict'](X) - y) ** 2))
        mse100 = float(np.mean((r100['predict'](X) - y) ** 2))
        assert mse100 <= mse10

    def test_predict_finite(self, regressionData):
        X, y = regressionData
        result = sq.ml.gradientBoosting(X, y, nEstimators=10)
        preds = result['predict'](X)
        assert np.all(np.isfinite(preds))

    def test_predict_on_new_data(self, regressionData):
        X, y = regressionData
        rng = np.random.default_rng(77)
        Xtest = rng.standard_normal((15, X.shape[1]))
        result = sq.ml.gradientBoosting(X, y, nEstimators=10)
        preds = result['predict'](Xtest)
        assert preds.shape == (15,)
        assert np.all(np.isfinite(preds))


# ---------------------------------------------------------------------------
# k-Means
# ---------------------------------------------------------------------------

class TestKMeans:
    def test_labels_shape(self, clusterData):
        X = clusterData
        result = sq.ml.kMeans(X, k=3, seed=0)
        assert result['labels'].shape == (X.shape[0],)

    def test_k_distinct_labels(self, clusterData):
        X = clusterData
        result = sq.ml.kMeans(X, k=3, seed=0)
        nDistinct = len(np.unique(result['labels']))
        assert nDistinct == 3

    def test_centroids_shape(self, clusterData):
        X = clusterData
        k = 3
        p = X.shape[1]
        result = sq.ml.kMeans(X, k=k, seed=0)
        assert result['centroids'].shape == (k, p)

    def test_inertia_positive(self, clusterData):
        result = sq.ml.kMeans(clusterData, k=3, seed=0)
        assert result['inertia'] > 0

    def test_nIter_positive_int(self, clusterData):
        result = sq.ml.kMeans(clusterData, k=3, seed=0)
        assert isinstance(result['nIter'], int)
        assert result['nIter'] >= 1

    def test_clusters_match_true_structure(self, clusterData):
        X = clusterData
        result = sq.ml.kMeans(X, k=3, seed=0)
        # The first 30, middle 30, last 30 should each have one dominant label
        labels = result['labels']
        for chunk in [labels[:30], labels[30:60], labels[60:]]:
            uniqueLabels, counts = np.unique(chunk, return_counts=True)
            dominantFraction = float(np.max(counts)) / len(chunk)
            assert dominantFraction >= 0.8  # at least 80% same label per cluster

    def test_inertia_decreases_with_more_clusters(self, clusterData):
        X = clusterData
        r3 = sq.ml.kMeans(X, k=3, seed=0)
        r6 = sq.ml.kMeans(X, k=6, seed=0)
        assert r6['inertia'] <= r3['inertia']

    def test_reproducible_with_seed(self, clusterData):
        X = clusterData
        r1 = sq.ml.kMeans(X, k=3, seed=42)
        r2 = sq.ml.kMeans(X, k=3, seed=42)
        np.testing.assert_array_equal(r1['labels'], r2['labels'])
