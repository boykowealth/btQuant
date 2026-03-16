"""
sipQuant.portfolio — Portfolio optimisation.
Pure NumPy. No pandas, scipy, or external dependencies.

Functions
---------
meanVariance    : Mean-variance optimisation (min variance or target return).
efficientFrontier: Trace the efficient frontier.
hrp             : Hierarchical Risk Parity (Lopez de Prado).
riskParity      : Equal risk contribution via cyclic coordinate descent.
blackLitterman  : Black-Litterman posterior views.
maxSharpe       : Maximum Sharpe ratio portfolio.
minCvar         : Minimum CVaR portfolio (Rockafellar-Uryasev gradient descent).
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _projSimplex(w):
    """Project vector w onto the probability simplex (sum=1, w>=0).
    Duchi et al. (2008) O(n log n) algorithm.
    """
    n = len(w)
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1.0) / (rho + 1.0)
    return np.maximum(w - theta, 0.0)


def _portReturn(w, mu):
    return float(w @ mu)


def _portVol(w, cov):
    return float(np.sqrt(w @ cov @ w))


# ---------------------------------------------------------------------------
# Mean-Variance
# ---------------------------------------------------------------------------

def meanVariance(mu, cov, targetReturn=None, allowShort=False):
    """Mean-variance optimisation.

    Parameters
    ----------
    mu           : (n,) expected returns.
    cov          : (n x n) covariance matrix.
    targetReturn : float or None. If None, returns the global minimum-variance
                   portfolio.
    allowShort   : bool. If True, analytical Lagrange solution (unconstrained
                   except weights sum to 1). If False, gradient descent with
                   momentum projected onto the simplex.

    Returns
    -------
    dict: weights (n,), return (float), volatility (float), sharpe (float).
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)

    if allowShort:
        # Analytical Lagrange solution.
        # Minimise w'Cw  s.t.  1'w = 1  [and optionally mu'w = targetReturn].
        covInv = np.linalg.inv(cov)
        ones = np.ones(n)
        A = float(ones @ covInv @ ones)
        B = float(ones @ covInv @ mu)
        C = float(mu @ covInv @ mu)
        D = A * C - B * B

        if targetReturn is None:
            # Global min-variance
            w = covInv @ ones / A
        else:
            lam = (C - B * targetReturn) / D
            gam = (A * targetReturn - B) / D
            w = covInv @ (lam * ones + gam * mu)
    else:
        # Long-only: projected gradient descent with momentum.
        w = np.ones(n) / n
        lr = 2e-3
        momentum = 0.9
        velocity = np.zeros(n)
        maxIter = 5000

        for _ in range(maxIter):
            grad = 2.0 * cov @ w
            if targetReturn is not None:
                # Augmented objective: variance + penalty for return deviation
                pen = 200.0 * (w @ mu - targetReturn)
                grad -= pen * mu
            velocity = momentum * velocity + lr * grad
            w_new = _projSimplex(w - velocity)
            if np.max(np.abs(w_new - w)) < 1e-9:
                w = w_new
                break
            w = w_new

    ret = _portReturn(w, mu)
    vol = _portVol(w, cov)
    sharpe = ret / vol if vol > 1e-12 else 0.0

    return {
        'weights': w,
        'return': ret,
        'volatility': vol,
        'sharpe': sharpe,
    }


# ---------------------------------------------------------------------------
# Efficient Frontier
# ---------------------------------------------------------------------------

def efficientFrontier(mu, cov, nPoints=50, allowShort=False):
    """Trace the efficient frontier.

    Parameters
    ----------
    mu         : (n,) expected returns.
    cov        : (n x n) covariance matrix.
    nPoints    : int number of frontier points.
    allowShort : bool.

    Returns
    -------
    dict: returns (nPoints,), volatilities (nPoints,), weights (nPoints x n).
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)

    # Determine return range from min-variance to max-return portfolios.
    mvp = meanVariance(mu, cov, targetReturn=None, allowShort=allowShort)
    rMin = mvp['return']
    rMax = float(np.max(mu))
    targets = np.linspace(rMin, rMax, nPoints)

    rets = np.zeros(nPoints)
    vols = np.zeros(nPoints)
    weights = np.zeros((nPoints, n))

    for i, tr in enumerate(targets):
        res = meanVariance(mu, cov, targetReturn=float(tr), allowShort=allowShort)
        rets[i] = res['return']
        vols[i] = res['volatility']
        weights[i] = res['weights']

    return {
        'returns': rets,
        'volatilities': vols,
        'weights': weights,
    }


# ---------------------------------------------------------------------------
# Hierarchical Risk Parity
# ---------------------------------------------------------------------------

def _corrToDistance(corr):
    """Distance matrix D_ij = sqrt((1 - rho_ij) / 2)."""
    return np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))


def _singleLinkage(dist):
    """Single-linkage hierarchical clustering.

    Returns a list of merge records: (i, j, distance, size).
    Uses naive O(n^3) loop — pure NumPy, no scipy.
    """
    n = dist.shape[0]
    # Work with indices into the current cluster set.
    clusters = {i: [i] for i in range(n)}
    ids = list(range(n))
    linkage = []
    nextId = n

    # Copy distance to a working dict of dicts.
    d = {}
    for i in ids:
        for j in ids:
            if i < j:
                d[(i, j)] = dist[i, j]

    active = set(ids)

    for _ in range(n - 1):
        # Find minimum distance pair among active clusters.
        minDist = np.inf
        pair = None
        aList = sorted(active)
        for ii in range(len(aList)):
            for jj in range(ii + 1, len(aList)):
                ci, cj = aList[ii], aList[jj]
                key = (min(ci, cj), max(ci, cj))
                if key in d and d[key] < minDist:
                    minDist = d[key]
                    pair = (ci, cj)

        ci, cj = pair
        merged = clusters[ci] + clusters[cj]
        size = len(merged)
        linkage.append((ci, cj, minDist, size))
        clusters[nextId] = merged
        active.discard(ci)
        active.discard(cj)
        active.add(nextId)

        # Update distances for new cluster (single linkage = min).
        for ck in active:
            if ck == nextId:
                continue
            k1 = (min(ci, ck), max(ci, ck))
            k2 = (min(cj, ck), max(cj, ck))
            d1 = d.get(k1, np.inf)
            d2 = d.get(k2, np.inf)
            newKey = (min(nextId, ck), max(nextId, ck))
            d[newKey] = min(d1, d2)

        nextId += 1

    return linkage, clusters, nextId - 1


def _getOrder(clusters, rootId):
    """Recursively extract leaf order from cluster dict."""
    c = clusters[rootId]
    if len(c) == 1:
        return c
    return c  # leaves already stored in merge order


def _quasiDiag(linkage, n):
    """Quasi-diagonalisation: returns leaf order from hierarchical clustering."""
    # Build a tree and do an in-order traversal.
    children = {}
    for (ci, cj, dist, size) in linkage:
        pass  # just need the last root

    # Reconstruct via recursive in-order traversal using linkage list.
    # Node IDs >= n are internal; < n are leaves.
    left = {}
    right = {}
    for idx, (ci, cj, dist, size) in enumerate(linkage):
        nodeId = n + idx
        left[nodeId] = ci
        right[nodeId] = cj

    rootId = n + len(linkage) - 1

    def inOrder(nodeId):
        if nodeId < n:
            return [nodeId]
        return inOrder(left[nodeId]) + inOrder(right[nodeId])

    return inOrder(rootId)


def _recursiveBisection(cov, order):
    """Recursive bisection: allocate weights by inverse variance."""
    weights = np.ones(len(order))

    def _bisect(items, w):
        if len(items) <= 1:
            return
        mid = len(items) // 2
        left = items[:mid]
        right = items[mid:]

        # Compute cluster variances.
        def clusterVar(idx):
            subCov = cov[np.ix_(idx, idx)]
            subCovInv = np.linalg.inv(subCov + np.eye(len(idx)) * 1e-10)
            ones = np.ones(len(idx))
            w_ivp = subCovInv @ ones / (ones @ subCovInv @ ones)
            return float(w_ivp @ subCov @ w_ivp)

        vL = clusterVar(left)
        vR = clusterVar(right)
        total = vL + vR
        alphaL = 1.0 - vL / total if total > 0 else 0.5
        alphaR = 1.0 - alphaL

        for i in left:
            w[order.index(i)] *= alphaL
        for i in right:
            w[order.index(i)] *= alphaR

        _bisect(left, w)
        _bisect(right, w)

    _bisect(list(range(len(order))), weights)
    return weights


def hrp(returns):
    """Hierarchical Risk Parity (Lopez de Prado).

    Steps:
      1. Correlation matrix from returns.
      2. Distance D = sqrt((1 - rho) / 2).
      3. Single-linkage hierarchical clustering (pure NumPy).
      4. Quasi-diagonalisation by reordering.
      5. Recursive bisection.

    Parameters
    ----------
    returns : (T x n) return matrix.

    Returns
    -------
    dict: weights (n,), order (list of ints).
    """
    returns = np.asarray(returns, dtype=float)
    T, n = returns.shape

    cov = np.cov(returns.T)
    stdDev = np.sqrt(np.diag(cov))
    stdDev = np.where(stdDev < 1e-12, 1e-12, stdDev)
    corr = cov / np.outer(stdDev, stdDev)
    np.fill_diagonal(corr, 1.0)

    dist = _corrToDistance(corr)
    linkage, clusters, rootId = _singleLinkage(dist)
    order = _quasiDiag(linkage, n)

    # Reorder covariance.
    covOrdered = cov[np.ix_(order, order)]
    rawWeights = _recursiveBisection(covOrdered, list(range(len(order))))

    # Map back to original asset indices.
    w = np.zeros(n)
    for posIdx, assetIdx in enumerate(order):
        w[assetIdx] = rawWeights[posIdx]

    w = np.maximum(w, 0.0)
    total = w.sum()
    if total > 1e-12:
        w /= total

    return {'weights': w, 'order': order}


# ---------------------------------------------------------------------------
# Risk Parity
# ---------------------------------------------------------------------------

def riskParity(cov, maxIter=500, tol=1e-8):
    """Equal risk contribution via cyclic coordinate descent (Spinu).

    Parameters
    ----------
    cov     : (n x n) covariance matrix.
    maxIter : int.
    tol     : float convergence tolerance.

    Returns
    -------
    dict: weights (n,), riskContributions (n,).
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    x = np.ones(n) / n
    targetRc = 1.0 / n  # equal risk contribution

    # Solve unconstrained problem: min 0.5 * y^T C y - (1/n) sum log(y_i)
    # Gradient: g_i = (Cy)_i - 1 / (n * y_i)
    # Gradient descent with adaptive step; then normalise y to get weights.
    y = np.ones(n, dtype=float)
    lr = 1.0 / (np.max(np.abs(np.diag(cov))) * n + 1e-12)

    for _ in range(maxIter):
        Cy = cov @ y
        g = Cy - 1.0 / (n * y)
        yNew = y - lr * g
        yNew = np.maximum(yNew, 1e-10)
        if np.max(np.abs(yNew - y)) < tol:
            y = yNew
            break
        y = yNew

    x = y / y.sum()
    Cx = cov @ x
    portVar = float(x @ Cx)
    rc = x * Cx / (np.sqrt(portVar) + 1e-16)

    return {'weights': x, 'riskContributions': rc}


# ---------------------------------------------------------------------------
# Black-Litterman
# ---------------------------------------------------------------------------

def blackLitterman(mu, cov, P, Q, Omega=None, tau=0.05, rf=0.0):
    """Black-Litterman posterior.

    Parameters
    ----------
    mu    : (n,) equilibrium returns (prior).
    cov   : (n x n) covariance.
    P     : (k x n) pick matrix.
    Q     : (k,) view returns.
    Omega : (k x k) view uncertainty. If None, proportional to P @ (tau*cov) @ P'.
    tau   : float, scalar uncertainty of prior.
    rf    : float, risk-free rate.

    Returns
    -------
    dict: muBL (n,), covBL (n x n), weights (n,).
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    n = len(mu)

    priorCov = tau * cov

    if Omega is None:
        Omega = np.diag(np.diag(P @ priorCov @ P.T))

    # Posterior mean and covariance.
    M = priorCov @ P.T @ np.linalg.inv(P @ priorCov @ P.T + Omega)
    muBL = mu + M @ (Q - P @ mu)
    covBL = (1.0 + tau) * cov - M @ P @ priorCov

    # MV weights from BL posterior (unconstrained).
    covInv = np.linalg.inv(covBL + np.eye(n) * 1e-10)
    ones = np.ones(n)
    w = covInv @ (muBL - rf)
    wSum = float(ones @ w)
    if abs(wSum) > 1e-12:
        w /= wSum

    return {'muBL': muBL, 'covBL': covBL, 'weights': w}


# ---------------------------------------------------------------------------
# Max Sharpe
# ---------------------------------------------------------------------------

def maxSharpe(mu, cov, rf=0.0, allowShort=False):
    """Maximum Sharpe ratio portfolio.

    Parameters
    ----------
    mu         : (n,) expected returns.
    cov        : (n x n) covariance.
    rf         : float, risk-free rate.
    allowShort : bool.

    Returns
    -------
    dict: weights (n,), sharpe (float), return (float), volatility (float).
    """
    mu = np.asarray(mu, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu)
    excess = mu - rf

    if allowShort:
        # Analytical: w proportional to Sigma^{-1} (mu - rf), then normalise.
        covInv = np.linalg.inv(cov + np.eye(n) * 1e-10)
        w = covInv @ excess
        wSum = w.sum()
        if abs(wSum) > 1e-12:
            w /= wSum
    else:
        # Projected gradient ascent on Sharpe.
        w = np.ones(n) / n
        lr = 1e-3
        momentum = 0.9
        velocity = np.zeros(n)

        for _ in range(8000):
            portRet = float(w @ excess)
            portVar = float(w @ cov @ w)
            portVol = np.sqrt(max(portVar, 1e-16))

            # Gradient of Sharpe w.r.t. w.
            dRet = excess
            dVol = (cov @ w) / portVol
            grad = (dRet * portVol - portRet * dVol) / (portVar + 1e-16)

            velocity = momentum * velocity + lr * grad
            w_new = _projSimplex(w + velocity)
            if np.max(np.abs(w_new - w)) < 1e-9:
                w = w_new
                break
            w = w_new

    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 1e-12 else 0.0

    return {'weights': w, 'sharpe': sharpe, 'return': ret, 'volatility': vol}


# ---------------------------------------------------------------------------
# Minimum CVaR
# ---------------------------------------------------------------------------

def minCvar(returns, alpha=0.05, maxIter=500):
    """Minimum CVaR portfolio (Rockafellar-Uryasev gradient descent).

    Parameters
    ----------
    returns : (T x n) return matrix.
    alpha   : float, significance level (e.g. 0.05 for 95% CVaR).
    maxIter : int.

    Returns
    -------
    dict: weights (n,), cvar (float), var (float).
    """
    returns = np.asarray(returns, dtype=float)
    T, n = returns.shape

    w = np.ones(n) / n
    lr = 1e-3
    momentum = 0.9
    velocity = np.zeros(n)
    nTail = max(1, int(np.ceil(alpha * T)))

    for _ in range(maxIter):
        portRets = returns @ w          # (T,)
        losses = -portRets              # positive = loss

        # CVaR gradient: average gradient over tail losses.
        sortedIdx = np.argsort(losses)[::-1]
        tailIdx = sortedIdx[:nTail]
        grad = -returns[tailIdx].mean(axis=0)

        velocity = momentum * velocity + lr * grad
        w_new = _projSimplex(w - velocity)
        if np.max(np.abs(w_new - w)) < 1e-9:
            w = w_new
            break
        w = w_new

    portRets = returns @ w
    losses = np.sort(-portRets)[::-1]
    varVal = float(losses[nTail - 1])
    cvarVal = float(losses[:nTail].mean())

    return {'weights': w, 'cvar': cvarVal, 'var': varVal}
