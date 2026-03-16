"""
Tests for sipQuant.dimension
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
def syntheticData():
    rng = np.random.default_rng(0)
    n, p = 50, 8
    # Low-rank signal
    factors = rng.standard_normal((n, 3))
    loadings = rng.standard_normal((3, p))
    X = factors @ loadings + 0.1 * rng.standard_normal((n, p))
    return X


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class TestPCA:
    def test_scores_shape(self, syntheticData):
        X = syntheticData
        n = X.shape[0]
        result = sq.dimension.pca(X, nComponents=3)
        assert result['scores'].shape == (n, 3)

    def test_explained_ratios_sum_leq_one(self, syntheticData):
        result = sq.dimension.pca(syntheticData)
        ratios = result['explainedVarianceRatio']
        assert float(np.sum(ratios)) <= 1.0 + 1e-10

    def test_explained_ratios_non_negative(self, syntheticData):
        result = sq.dimension.pca(syntheticData)
        assert np.all(result['explainedVarianceRatio'] >= 0)

    def test_components_shape(self, syntheticData):
        X = syntheticData
        p = X.shape[1]
        result = sq.dimension.pca(X, nComponents=3)
        assert result['components'].shape == (3, p)

    def test_loadings_shape(self, syntheticData):
        X = syntheticData
        p = X.shape[1]
        result = sq.dimension.pca(X, nComponents=3)
        assert result['loadings'].shape == (p, 3)

    def test_nComponents_stored(self, syntheticData):
        result = sq.dimension.pca(syntheticData, nComponents=4)
        assert result['nComponents'] == 4

    def test_default_nComponents_is_min_n_p(self, syntheticData):
        X = syntheticData
        n, p = X.shape
        result = sq.dimension.pca(X)
        assert result['nComponents'] == min(n, p)

    def test_first_component_max_variance(self, syntheticData):
        result = sq.dimension.pca(syntheticData, nComponents=3)
        ev = result['explainedVariance']
        # Variance should be descending
        assert ev[0] >= ev[1] >= ev[2]

    def test_scores_zero_mean_when_centered(self, syntheticData):
        result = sq.dimension.pca(syntheticData, nComponents=2, center=True)
        colMeans = np.mean(result['scores'], axis=0)
        np.testing.assert_allclose(colMeans, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Kernel PCA
# ---------------------------------------------------------------------------

class TestKernelPCA:
    def test_scores_shape(self, syntheticData):
        X = syntheticData
        n = X.shape[0]
        result = sq.dimension.kernelPca(X, nComponents=2)
        assert result['scores'].shape == (n, 2)

    def test_eigenvalues_length(self, syntheticData):
        result = sq.dimension.kernelPca(syntheticData, nComponents=2)
        assert len(result['eigenvalues']) == 2

    def test_eigenvalues_non_negative(self, syntheticData):
        result = sq.dimension.kernelPca(syntheticData, nComponents=2)
        # Top eigenvalues should be non-negative (kernel PCA uses eigh, real eigs)
        assert np.all(result['eigenvalues'] >= -1e-10)

    def test_scores_finite(self, syntheticData):
        result = sq.dimension.kernelPca(syntheticData, nComponents=2)
        assert np.all(np.isfinite(result['scores']))

    def test_different_gamma(self, syntheticData):
        r1 = sq.dimension.kernelPca(syntheticData, nComponents=2, gamma=0.1)
        r2 = sq.dimension.kernelPca(syntheticData, nComponents=2, gamma=10.0)
        # Different gamma should produce different scores
        assert not np.allclose(r1['scores'], r2['scores'])


# ---------------------------------------------------------------------------
# ICA
# ---------------------------------------------------------------------------

class TestICA:
    def setup_method(self):
        rng = np.random.default_rng(1)
        n = 80
        # Two independent sources: uniform and Laplace-like
        s1 = rng.uniform(-1, 1, n)
        s2 = np.sign(rng.standard_normal(n)) * rng.exponential(1, n)
        S = np.column_stack([s1, s2])
        A = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.X = S @ A.T

    def test_sources_shape(self):
        result = sq.dimension.ica(self.X, nComponents=2)
        assert result['sources'].shape == (self.X.shape[0], 2)

    def test_components_shape(self):
        result = sq.dimension.ica(self.X, nComponents=2)
        assert result['components'].shape == (2, self.X.shape[1])

    def test_nComponents_stored(self):
        result = sq.dimension.ica(self.X, nComponents=2)
        assert result['nComponents'] == 2

    def test_sources_finite(self):
        result = sq.dimension.ica(self.X, nComponents=2)
        assert np.all(np.isfinite(result['sources']))

    def test_default_nComponents(self, syntheticData):
        result = sq.dimension.ica(syntheticData)
        n, p = syntheticData.shape
        assert result['nComponents'] == min(n, p)


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

class TestTSNE:
    def setup_method(self):
        rng = np.random.default_rng(7)
        # Three well-separated clusters in 5D
        c1 = rng.standard_normal((15, 5)) + np.array([5, 0, 0, 0, 0])
        c2 = rng.standard_normal((15, 5)) + np.array([-5, 0, 0, 0, 0])
        c3 = rng.standard_normal((15, 5)) + np.array([0, 5, 0, 0, 0])
        self.X = np.vstack([c1, c2, c3])

    def test_embedding_shape(self):
        result = sq.dimension.tsne(self.X, nComponents=2, nIter=100, seed=0)
        assert result['embedding'].shape == (45, 2)

    def test_embedding_finite(self):
        result = sq.dimension.tsne(self.X, nComponents=2, nIter=100, seed=0)
        assert np.all(np.isfinite(result['embedding']))

    def test_embedding_not_all_zeros(self):
        result = sq.dimension.tsne(self.X, nComponents=2, nIter=100, seed=0)
        assert np.std(result['embedding']) > 0

    def test_embedding_key_exists(self):
        result = sq.dimension.tsne(self.X, nComponents=2, nIter=50, seed=1)
        assert 'embedding' in result

    def test_reproducible_with_seed(self):
        r1 = sq.dimension.tsne(self.X, nComponents=2, nIter=50, seed=42)
        r2 = sq.dimension.tsne(self.X, nComponents=2, nIter=50, seed=42)
        np.testing.assert_array_equal(r1['embedding'], r2['embedding'])
