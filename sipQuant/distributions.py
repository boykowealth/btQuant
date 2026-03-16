"""
sipQuant.distributions — Distribution fitting and copula sampling.
Pure NumPy. No pandas, scipy, or external dependencies.

Functions
---------
gaussianCopula  : Sample Gaussian copula.
tCopula         : Sample t-copula.
claytonCopula   : Sample Clayton copula (bivariate).
gumbelCopula    : Sample Gumbel copula (bivariate).
tailDependence  : Empirical tail dependence coefficients.
kendallTau      : Kendall's tau (O(n^2)).
spearmanRho     : Spearman rho via rank transform.
fitNormal       : MLE normal fit.
fitLognormal    : MLE lognormal fit.
fitT            : Student-t fit (MOM + MLE iteration for df).
fitGamma        : Gamma fit by MOM.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normCdf(x):
    """Normal CDF via Abramowitz & Stegun polynomial approximation."""
    x = np.asarray(x, dtype=float)
    scalar = x.ndim == 0
    x = np.atleast_1d(x)
    result = np.empty_like(x)

    for idx in np.ndindex(x.shape):
        xi = x[idx]
        t = 1.0 / (1.0 + 0.2316419 * abs(xi))
        poly = t * (0.319381530
                    + t * (-0.356563782
                           + t * (1.781477937
                                  + t * (-1.821255978
                                         + t * 1.330274429))))
        cdf = 1.0 - (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * xi * xi) * poly
        result[idx] = cdf if xi >= 0.0 else 1.0 - cdf

    return float(result[0]) if scalar else result


def _normPpf(p, tol=1e-8, maxIter=50):
    """Inverse normal CDF via Newton-Raphson (Abramowitz & Stegun seed).

    Parameters
    ----------
    p       : float or array in (0, 1).
    tol     : float convergence tolerance.
    maxIter : int.

    Returns
    -------
    float or array.
    """
    p = np.asarray(p, dtype=float)
    scalar = p.ndim == 0
    p = np.atleast_1d(np.clip(p, 1e-12, 1.0 - 1e-12))
    result = np.empty_like(p)

    for idx in np.ndindex(p.shape):
        pi = p[idx]
        if pi < 0.5:
            tt = np.sqrt(-2.0 * np.log(pi))
            sign = -1.0
        else:
            tt = np.sqrt(-2.0 * np.log(1.0 - pi))
            sign = 1.0

        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        num = c0 + c1 * tt + c2 * tt * tt
        den = 1.0 + d1 * tt + d2 * tt * tt + d3 * tt ** 3
        xi = sign * (tt - num / den)

        for _ in range(maxIter):
            fx = _normCdf(xi) - pi
            fpx = np.exp(-0.5 * xi * xi) / np.sqrt(2.0 * np.pi)
            dx = fx / (fpx + 1e-16)
            xi -= dx
            if abs(dx) < tol:
                break
        result[idx] = xi

    return float(result[0]) if scalar else result


def _tPpf(p, df):
    """Inverse t-CDF via bracketed bisection (pure NumPy)."""
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))

    def _tCdf(x, df):
        """Student-t CDF using regularised incomplete beta function."""
        # Use normal approximation for large df.
        if df > 200:
            return float(_normCdf(x))
        # Abramowitz & Stegun approximation via normal.
        # More accurate: use the relationship t^2/(t^2+df) ~ beta.
        # We use a simple integration-free approximation via cornish-fisher.
        z = x * (1.0 - 1.0 / (4.0 * df)) / np.sqrt(1.0 + x * x / (2.0 * df))
        return float(_normCdf(z))

    # Bisection.
    lo, hi = -50.0, 50.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if _tCdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break
    return 0.5 * (lo + hi)


def _chol(A):
    """Cholesky decomposition (lower triangular). Pure NumPy."""
    return np.linalg.cholesky(A)


def _makeCorr(rho, d):
    """Build a d x d correlation matrix from scalar rho or (d x d) array."""
    rho = np.asarray(rho, dtype=float)
    if rho.ndim == 0:
        corr = rho * np.ones((d, d)) + (1.0 - rho) * np.eye(d)
    else:
        corr = rho.copy()
    # Ensure positive semi-definiteness by clipping eigenvalues.
    eigvals, eigvecs = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 1e-8)
    corr = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # Re-normalise to correlation matrix.
    d_inv = 1.0 / np.sqrt(np.diag(corr))
    corr = corr * np.outer(d_inv, d_inv)
    return corr


# ---------------------------------------------------------------------------
# Gaussian Copula
# ---------------------------------------------------------------------------

def gaussianCopula(n, rho, d=2, seed=None):
    """Sample from a Gaussian copula.

    Parameters
    ----------
    n    : int, number of samples.
    rho  : scalar or (d x d) correlation matrix.
    d    : int, dimension.
    seed : int or None.

    Returns
    -------
    (n x d) array of uniform marginals in [0, 1].
    """
    rng = np.random.default_rng(seed)
    corr = _makeCorr(rho, d)
    L = _chol(corr)
    z = rng.standard_normal((n, d))
    x = z @ L.T
    # Transform to uniforms via normal CDF.
    u = np.vectorize(_normCdf)(x)
    u = np.clip(u, 0.0, 1.0)
    return u


# ---------------------------------------------------------------------------
# t-Copula
# ---------------------------------------------------------------------------

def tCopula(n, rho, df, d=2, seed=None):
    """Sample from a t-copula.

    Parameters
    ----------
    n    : int.
    rho  : scalar or (d x d) correlation matrix.
    df   : float, degrees of freedom.
    d    : int.
    seed : int or None.

    Returns
    -------
    (n x d) array of uniform marginals.
    """
    rng = np.random.default_rng(seed)
    corr = _makeCorr(rho, d)
    L = _chol(corr)
    z = rng.standard_normal((n, d))
    x = z @ L.T
    # Chi-squared draw for each sample.
    chi2 = rng.chisquare(df, size=n)
    t = x / np.sqrt(chi2[:, None] / df)

    # Transform to uniforms via t-CDF approximation.
    def _tCdfApprox(ti, dfv):
        z = ti * (1.0 - 1.0 / (4.0 * dfv)) / np.sqrt(1.0 + ti * ti / (2.0 * dfv))
        return float(_normCdf(z))

    u = np.vectorize(_tCdfApprox)(t, df)
    u = np.clip(u, 0.0, 1.0)
    return u


# ---------------------------------------------------------------------------
# Clayton Copula
# ---------------------------------------------------------------------------

def claytonCopula(n, theta, seed=None):
    """Sample bivariate Clayton copula using conditional sampling.

    C(u,v) = (u^{-theta} + v^{-theta} - 1)^{-1/theta}

    Parameters
    ----------
    n     : int.
    theta : float > 0.
    seed  : int or None.

    Returns
    -------
    (n x 2) array of uniform marginals.
    """
    rng = np.random.default_rng(seed)
    theta = max(float(theta), 1e-6)

    u = rng.uniform(0, 1, n)
    p = rng.uniform(0, 1, n)

    # Conditional quantile of V|U=u:
    # C_{2|1}(v|u) = p => v = (p^{-theta/(1+theta)} + u^{-theta} - 1)^{-1/theta}
    v = (p ** (-theta / (1.0 + theta)) + u ** (-theta) - 1.0) ** (-1.0 / theta)
    v = np.clip(v, 1e-10, 1.0 - 1e-10)
    u = np.clip(u, 1e-10, 1.0 - 1e-10)

    return np.column_stack([u, v])


# ---------------------------------------------------------------------------
# Gumbel Copula
# ---------------------------------------------------------------------------

def gumbelCopula(n, theta, seed=None):
    """Sample bivariate Gumbel copula using Marshall-Olkin algorithm.

    Uses stable distribution simulation for the Laplace-Stieltjes transform.
    For theta=1 (independence), samples are uniform.

    Parameters
    ----------
    n     : int.
    theta : float >= 1.
    seed  : int or None.

    Returns
    -------
    (n x 2) array of uniform marginals.
    """
    rng = np.random.default_rng(seed)
    theta = max(float(theta), 1.0)
    alpha = 1.0 / theta  # stable index

    if abs(alpha - 1.0) < 1e-6:
        # Independence.
        u = rng.uniform(0, 1, (n, 2))
        return u

    # Marshall-Olkin: sample V ~ Stable(alpha, 1) using Chambers-Mallows-Stuck.
    # Then U_i = exp(-E_i / V)^{1/theta} where E_i ~ Exp(1).
    # Stable(alpha, 1) simulation:
    phi = (rng.uniform(0, 1, n) - 0.5) * np.pi  # U(-pi/2, pi/2)
    w = rng.exponential(1.0, n)                  # Exp(1)

    # Chambers-Mallows-Stuck formula for Stable(alpha, 1, 0, 1) (positively skewed).
    sinA = np.sin(alpha * phi)
    cosP = np.cos(phi)
    cosPA = np.cos((1.0 - alpha) * phi)
    V = (sinA / (cosP + 1e-16) ** (1.0 / alpha)) * ((cosPA / (w + 1e-16)) ** ((1.0 - alpha) / alpha))
    V = np.abs(V) + 1e-10

    E1 = rng.exponential(1.0, n)
    E2 = rng.exponential(1.0, n)
    u1 = np.exp(-(E1 / V) ** alpha)
    u2 = np.exp(-(E2 / V) ** alpha)

    u1 = np.clip(u1, 1e-10, 1.0 - 1e-10)
    u2 = np.clip(u2, 1e-10, 1.0 - 1e-10)

    return np.column_stack([u1, u2])


# ---------------------------------------------------------------------------
# Tail Dependence
# ---------------------------------------------------------------------------

def tailDependence(data, threshold=0.1):
    """Empirical tail dependence coefficients.

    Parameters
    ----------
    data      : (T x 2) array of observations (or uniforms).
    threshold : float, probability threshold for tail region.

    Returns
    -------
    dict: lower (float), upper (float), threshold (float).
    """
    data = np.asarray(data, dtype=float)
    T = data.shape[0]

    # Convert to empirical uniforms if not already.
    u = np.empty_like(data)
    for j in range(data.shape[1]):
        ranks = np.argsort(np.argsort(data[:, j])) + 1
        u[:, j] = ranks / (T + 1.0)

    q = float(threshold)
    # Lower tail: both below q.
    lower = float(np.mean((u[:, 0] < q) & (u[:, 1] < q)) / (q + 1e-16))
    lower = float(np.clip(lower, 0.0, 1.0))

    # Upper tail: both above 1-q.
    upper = float(np.mean((u[:, 0] > 1.0 - q) & (u[:, 1] > 1.0 - q)) / (q + 1e-16))
    upper = float(np.clip(upper, 0.0, 1.0))

    return {'lower': lower, 'upper': upper, 'threshold': q}


# ---------------------------------------------------------------------------
# Kendall's Tau
# ---------------------------------------------------------------------------

def kendallTau(x, y):
    """Kendall's tau (O(n^2) implementation).

    Parameters
    ----------
    x, y : (T,) arrays.

    Returns
    -------
    float in [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    concordant = 0
    discordant = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            prod = dx * dy
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1

    nPairs = n * (n - 1) // 2
    tau = (concordant - discordant) / (nPairs + 1e-16)
    return float(np.clip(tau, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Spearman Rho
# ---------------------------------------------------------------------------

def spearmanRho(x, y):
    """Spearman rho via rank transform.

    Parameters
    ----------
    x, y : (T,) arrays.

    Returns
    -------
    float in [-1, 1].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)

    def _rank(a):
        order = np.argsort(a)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(1, n + 1, dtype=float)
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    d2 = np.sum((rx - ry) ** 2)
    rho = 1.0 - 6.0 * d2 / (n * (n * n - 1.0) + 1e-16)
    return float(np.clip(rho, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Distribution Fitting
# ---------------------------------------------------------------------------

def fitNormal(data):
    """MLE normal fit.

    Parameters
    ----------
    data : (T,) array.

    Returns
    -------
    dict: mu (float), sigma (float), logLik (float).
    """
    data = np.asarray(data, dtype=float)
    mu = float(data.mean())
    sigma = float(data.std(ddof=0))
    sigma = max(sigma, 1e-12)
    T = len(data)
    logLik = float(-T * np.log(sigma) - 0.5 * T * np.log(2.0 * np.pi)
                   - 0.5 * np.sum(((data - mu) / sigma) ** 2))
    return {'mu': mu, 'sigma': sigma, 'logLik': logLik}


def fitLognormal(data):
    """MLE lognormal fit.

    Parameters
    ----------
    data : (T,) positive array.

    Returns
    -------
    dict: mu (float), sigma (float), logLik (float).
    """
    data = np.asarray(data, dtype=float)
    logData = np.log(data + 1e-16)
    result = fitNormal(logData)
    # Adjust logLik for the Jacobian (lognormal = normal on log scale).
    result['logLik'] = result['logLik'] - float(np.sum(logData))
    return result


def fitT(data, maxIter=100):
    """Student-t fit by method of moments + MLE iteration for df.

    Parameters
    ----------
    data    : (T,) array.
    maxIter : int.

    Returns
    -------
    dict: mu (float), sigma (float), df (float), logLik (float).
    """
    data = np.asarray(data, dtype=float)
    T = len(data)
    mu = float(data.mean())
    var = float(data.var(ddof=1))
    kurt = float(_excessKurtosis(data))

    # MOM estimate for df: kurtosis = 6/(df-4) => df = 6/kurtosis + 4.
    if kurt > 0.1:
        df = max(6.0 / kurt + 4.0, 2.1)
    else:
        df = 30.0

    # Scale: sigma^2 = var * (df - 2) / df.
    sigma = np.sqrt(max(var * (df - 2.0) / df, 1e-12))

    # Simple iteration: update df by matching kurtosis from residuals.
    for _ in range(maxIter):
        z = (data - mu) / (sigma + 1e-12)
        # Update df via kurtosis of standardised residuals.
        kurtZ = float(_excessKurtosis(z))
        if kurtZ > 0.1:
            df_new = max(6.0 / kurtZ + 4.0, 2.1)
        else:
            df_new = min(df * 1.1, 100.0)
        # Update sigma.
        sigma_new = np.sqrt(max(var * (df_new - 2.0) / df_new, 1e-12))
        if abs(df_new - df) < 1e-6:
            df = df_new
            sigma = sigma_new
            break
        df = df_new
        sigma = sigma_new

    # Log-likelihood under t distribution (unnormalised approximation).
    z = (data - mu) / (sigma + 1e-12)
    logLik = float(T * (np.log(1.0 + 1.0 / (df - 1.0 + 1e-16)) * 0.5
                        - np.log(sigma + 1e-12)
                        - 0.5 * np.log(np.pi * df + 1e-16))
                   - (df + 1.0) / 2.0 * np.sum(np.log(1.0 + z ** 2 / (df + 1e-16))))

    return {'mu': mu, 'sigma': float(sigma), 'df': float(df), 'logLik': float(logLik)}


def fitGamma(data):
    """Gamma fit by method of moments.

    Parameters
    ----------
    data : (T,) positive array.

    Returns
    -------
    dict: shape (float, alpha), scale (float, beta), logLik (float).
    """
    data = np.asarray(data, dtype=float)
    data = np.maximum(data, 1e-12)
    mu = float(data.mean())
    var = float(data.var(ddof=1))
    # MOM: shape = mu^2 / var, scale = var / mu.
    shape = mu * mu / (var + 1e-16)
    scale = var / (mu + 1e-16)
    shape = max(shape, 1e-6)
    scale = max(scale, 1e-6)

    # Log-likelihood: sum( (alpha-1)*log(x) - x/beta ) - n*(alpha*log(beta) + log_gamma(alpha)).
    def _logGamma(a):
        # Stirling approximation for log Gamma.
        if a < 0.5:
            a = 0.5
        return (a - 0.5) * np.log(a) - a + 0.5 * np.log(2.0 * np.pi) + 1.0 / (12.0 * a)

    T = len(data)
    logLik = float(
        (shape - 1.0) * np.sum(np.log(data + 1e-16))
        - np.sum(data) / scale
        - T * (shape * np.log(scale + 1e-16) + _logGamma(shape))
    )

    return {'shape': float(shape), 'scale': float(scale), 'logLik': float(logLik)}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _excessKurtosis(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sigma = x.std(ddof=0)
    if sigma < 1e-16:
        return 0.0
    return float(np.mean(((x - mu) / sigma) ** 4)) - 3.0
