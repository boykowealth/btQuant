"""
sipQuant.dimension — Dimensionality reduction methods.
SIP Global (Systematic Index Partners)

Pure NumPy. No external dependencies.

Functions
---------
pca       : Principal Component Analysis via SVD.
kernelPca : RBF kernel PCA.
ica       : FastICA (deflation).
tsne      : t-SNE embedding.
"""

import numpy as np


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

def pca(X, nComponents=None, center=True, scale=False):
    """Principal Component Analysis using SVD.

    Parameters
    ----------
    X           : array-like, shape (n, p) — input data.
    nComponents : int or None — number of components to return (default: all).
    center      : bool — subtract column means (default True).
    scale       : bool — divide by column std deviations (default False).

    Returns
    -------
    dict with keys:
        components          — (nComponents, p) principal component directions.
        explainedVariance   — (nComponents,) variance explained by each component.
        explainedVarianceRatio — (nComponents,) fraction of total variance.
        scores              — (n, nComponents) projected data.
        loadings            — (p, nComponents) loadings matrix.
        nComponents         — int.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if nComponents is None:
        nComponents = min(n, p)
    nComponents = int(nComponents)

    Xc = X.copy()
    if center:
        Xc -= np.mean(Xc, axis=0)
    if scale:
        stds = np.std(Xc, axis=0, ddof=1)
        stds[stds == 0] = 1.0
        Xc /= stds

    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)

    explainedVariance = (s ** 2) / (n - 1)
    totalVariance = np.sum(explainedVariance)
    explainedVarianceRatio = explainedVariance / totalVariance if totalVariance > 0 else explainedVariance

    components = Vt[:nComponents]
    scores = U[:, :nComponents] * s[:nComponents]
    loadings = Vt[:nComponents].T * s[:nComponents] / np.sqrt(n - 1)

    return {
        'components': components,
        'explainedVariance': explainedVariance[:nComponents],
        'explainedVarianceRatio': explainedVarianceRatio[:nComponents],
        'scores': scores,
        'loadings': loadings,
        'nComponents': nComponents,
    }


# ---------------------------------------------------------------------------
# Kernel PCA
# ---------------------------------------------------------------------------

def kernelPca(X, nComponents=2, kernel='rbf', gamma=1.0):
    """RBF kernel PCA.

    K_ij = exp(-gamma * ||x_i - x_j||^2)
    Centre the kernel matrix, then eigendecompose.

    Parameters
    ----------
    X           : array-like, shape (n, p).
    nComponents : int — number of kernel PCA components (default 2).
    kernel      : str — only 'rbf' is supported.
    gamma       : float — RBF bandwidth parameter.

    Returns
    -------
    dict with keys: scores (n x nComponents), eigenvalues (nComponents,).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    nComponents = int(nComponents)
    gamma = float(gamma)

    # Pairwise squared distances
    sq = np.sum(X ** 2, axis=1)
    distSq = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    distSq = np.maximum(distSq, 0.0)

    K = np.exp(-gamma * distSq)

    # Centre kernel matrix: K_c = (I - 1/n * 11^T) K (I - 1/n * 11^T)
    oneN = np.ones((n, n)) / n
    Kc = K - oneN @ K - K @ oneN + oneN @ K @ oneN

    eigVals, eigVecs = np.linalg.eigh(Kc)

    # Sort descending
    idx = np.argsort(eigVals)[::-1]
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:, idx]

    eigVals = eigVals[:nComponents]
    eigVecs = eigVecs[:, :nComponents]

    # Normalise eigenvectors by sqrt of eigenvalue
    sqrtEig = np.sqrt(np.maximum(eigVals, 1e-12))
    scores = eigVecs * sqrtEig[None, :]

    return {
        'scores': scores,
        'eigenvalues': eigVals,
    }


# ---------------------------------------------------------------------------
# ICA
# ---------------------------------------------------------------------------

def ica(X, nComponents=None, maxIter=200, tol=1e-4):
    """FastICA using deflation algorithm.

    Whitens X, then for each component iterates:
        w <- E[X * g(w^T X)] - E[g'(w^T X)] * w
        w /= ||w||
    where g(u) = tanh(u) (logistic non-linearity).

    Parameters
    ----------
    X           : array-like, shape (n, p).
    nComponents : int or None — number of independent components (default: min(n, p)).
    maxIter     : int — maximum iterations per component.
    tol         : float — convergence tolerance.

    Returns
    -------
    dict with keys: components (nComponents x p), sources (n x nComponents), nComponents.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape

    if nComponents is None:
        nComponents = min(n, p)
    nComponents = int(nComponents)

    # Centre
    Xc = X - np.mean(X, axis=0)

    # Whiten: PCA whitening
    cov = Xc.T @ Xc / (n - 1)
    eigVals, eigVecs = np.linalg.eigh(cov)
    idx = np.argsort(eigVals)[::-1]
    eigVals = eigVals[idx]
    eigVecs = eigVecs[:, idx]

    eigVals = eigVals[:nComponents]
    eigVecs = eigVecs[:, :nComponents]

    sqrtInvEig = 1.0 / np.sqrt(np.maximum(eigVals, 1e-12))
    Xwhite = Xc @ eigVecs * sqrtInvEig[None, :]  # (n, nComponents)

    rng = np.random.default_rng(42)
    W = np.zeros((nComponents, nComponents))

    for c in range(nComponents):
        w = rng.standard_normal(nComponents)
        w /= np.linalg.norm(w)

        for _ in range(maxIter):
            wX = Xwhite @ w  # (n,)
            gWx = np.tanh(wX)
            gPrimeWx = 1.0 - np.tanh(wX) ** 2

            wNew = (Xwhite.T @ gWx) / n - np.mean(gPrimeWx) * w

            # Deflation: subtract projections onto previous components
            for j in range(c):
                wNew -= (wNew @ W[j]) * W[j]

            norm = np.linalg.norm(wNew)
            if norm < 1e-15:
                break
            wNew /= norm

            delta = np.abs(np.abs(wNew @ w) - 1.0)
            w = wNew
            if delta < tol:
                break

        W[c] = w

    # Unmixing matrix maps whitened data to sources
    sources = Xwhite @ W.T  # (n, nComponents)
    # Components in original space
    components = W @ (eigVecs * sqrtInvEig[None, :]).T  # (nComponents, p)

    return {
        'components': components,
        'sources': sources,
        'nComponents': nComponents,
    }


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

def tsne(X, nComponents=2, perplexity=30.0, nIter=500, lr=200.0, seed=None):
    """t-SNE dimensionality reduction.

    Computes pairwise affinities in high-dimensional space using a Gaussian
    kernel calibrated to perplexity, then optimises a low-dimensional
    embedding by minimising KL divergence using Student-t affinities.

    Parameters
    ----------
    X          : array-like, shape (n, p).
    nComponents: int — embedding dimension (default 2).
    perplexity : float — effective number of neighbours (default 30).
    nIter      : int — gradient descent iterations (default 500).
    lr         : float — learning rate (default 200).
    seed       : int or None — RNG seed.

    Returns
    -------
    dict with key: embedding (n x nComponents).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    nComponents = int(nComponents)
    perplexity = float(perplexity)

    rng = np.random.default_rng(seed)

    # --- Pairwise squared distances in high-dim space ---
    sq = np.sum(X ** 2, axis=1)
    distSq = sq[:, None] + sq[None, :] - 2.0 * X @ X.T
    distSq = np.maximum(distSq, 0.0)
    np.fill_diagonal(distSq, 0.0)

    # --- Compute conditional probabilities (binary search for sigma) ---
    logPerp = np.log(perplexity)
    P = np.zeros((n, n))

    for i in range(n):
        betaMin = -np.inf
        betaMax = np.inf
        beta = 1.0
        diSq = distSq[i].copy()
        diSq[i] = np.inf  # exclude self

        for _ in range(50):
            expD = np.exp(-beta * diSq)
            expD[i] = 0.0
            sumExpD = np.sum(expD)
            if sumExpD == 0:
                sumExpD = 1e-12
            H = np.log(sumExpD) + beta * np.sum(diSq * expD) / sumExpD
            Hdiff = H - logPerp
            if abs(Hdiff) < 1e-5:
                break
            if Hdiff > 0:
                betaMin = beta
                beta = beta * 2.0 if betaMax == np.inf else (beta + betaMax) / 2.0
            else:
                betaMax = beta
                beta = beta / 2.0 if betaMin == -np.inf else (beta + betaMin) / 2.0

        P[i] = expD / sumExpD

    # Symmetrise and normalise
    P = (P + P.T) / (2.0 * n)
    P = np.maximum(P, 1e-12)

    # Early exaggeration
    P *= 4.0

    # --- Initialise embedding ---
    Y = rng.standard_normal((n, nComponents)) * 0.01
    velocity = np.zeros_like(Y)
    gains = np.ones_like(Y)
    momentum = 0.5

    for iteration in range(nIter):
        if iteration == 100:
            P /= 4.0  # remove early exaggeration
        if iteration == 250:
            momentum = 0.8

        # Student-t affinities in low-dim space
        sqY = np.sum(Y ** 2, axis=1)
        distSqY = sqY[:, None] + sqY[None, :] - 2.0 * Y @ Y.T
        distSqY = np.maximum(distSqY, 0.0)
        np.fill_diagonal(distSqY, 0.0)

        num = 1.0 / (1.0 + distSqY)
        np.fill_diagonal(num, 0.0)
        Q = num / np.maximum(np.sum(num), 1e-12)
        Q = np.maximum(Q, 1e-12)

        # Gradient
        PQ = P - Q
        grad = np.zeros_like(Y)
        for d in range(nComponents):
            diff = Y[:, d:d+1] - Y[:, d]  # (n, n) differences
            grad[:, d] = 4.0 * np.sum(PQ * num * diff, axis=1)

        # Adaptive learning rate (gains)
        gainUpdate = ((grad > 0) != (velocity > 0)).astype(float) * 0.2
        gainUpdate += ((grad > 0) == (velocity > 0)).astype(float) * 0.8
        gains = np.maximum(gains * gainUpdate + (1 - gainUpdate) * 0.01, 0.01)

        velocity = momentum * velocity - lr * gains * grad
        Y += velocity
        Y -= np.mean(Y, axis=0)

    return {'embedding': Y}
