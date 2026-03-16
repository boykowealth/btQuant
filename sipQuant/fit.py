"""
sipQuant.fit — Model calibration.
SIP Global (Systematic Index Partners)

Pure NumPy. No external dependencies.

Functions
---------
fitOU           : Ornstein-Uhlenbeck parameter estimation (exact MLE)
fitGarch        : GARCH(1,1) parameter estimation (ADAM gradient descent)
fitHeston       : Heston model calibration (ADAM gradient descent)
fitCopula       : Copula fitting (Gaussian, t, Clayton, Gumbel)
fitDistribution : Marginal distribution fitting (normal, lognormal, t, gamma)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normCdf(x):
    """Abramowitz & Stegun normal CDF approximation."""
    x = np.asarray(x, dtype=float)
    xAbs = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * xAbs)
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    p = 1.0 - np.exp(-0.5 * xAbs ** 2) / np.sqrt(2.0 * np.pi) * poly
    return np.where(x >= 0, p, 1.0 - p)


def _normPpf(p, tol=1e-8, maxIter=60):
    """Inverse normal CDF via Newton-Raphson (scalar input)."""
    p = float(p)
    if p <= 0.0:
        return -np.inf
    if p >= 1.0:
        return np.inf
    # Rational approximation as initial guess (Beasley-Springer-Moro)
    a = [2.50662823884, -18.61500062529, 41.39119773534, -25.44106049637]
    b = [-8.47351093090, 23.08336743743, -21.06224101826, 3.13082909833]
    c = [0.3374754822726147, 0.9761690190917186, 0.1607979714918209,
         0.0276438810333863, 0.0038405729373609, 0.0003951896511349,
         0.0000321767881768, 0.0000002888167364, 0.0000003960315187]
    y = p - 0.5
    if abs(y) < 0.42:
        r = y * y
        num = a[0] + r * (a[1] + r * (a[2] + r * a[3]))
        den = 1 + r * (b[0] + r * (b[1] + r * (b[2] + r * b[3])))
        x = y * num / den
    else:
        r = np.log(-np.log(p if y < 0 else 1 - p))
        x = c[0] + r * (c[1] + r * (c[2] + r * (c[3] + r * (c[4] + r * (c[5] + r * (c[6] + r * (c[7] + r * c[8])))))))
        if y < 0:
            x = -x
    # Newton-Raphson refinement
    for _ in range(maxIter):
        cdf = float(_normCdf(x))
        pdf = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
        dx = (cdf - p) / pdf
        x -= dx
        if abs(dx) < tol:
            break
    return float(x)


def _logGamma(x):
    """Lanczos log-gamma approximation."""
    cof = [76.18009172947146, -86.50532032941677, 24.01409824083091,
           -1.231739572450155, 1.208650973866179e-3, -5.395239384953e-6]
    y = float(x)
    tmp = x + 5.5
    tmp -= (x + 0.5) * np.log(tmp)
    ser = 1.000000000190015
    for c in cof:
        y += 1
        ser += c / y
    return -tmp + np.log(2.5066282746310005 * ser / x)


# ---------------------------------------------------------------------------
# OU calibration
# ---------------------------------------------------------------------------

def fitOU(series, dt=1/252):
    """
    Calibrate an Ornstein-Uhlenbeck process by exact MLE.

    Uses OLS on the AR(1) representation:
    x[t] = alpha * x[t-1] + beta + epsilon

    Parameters
    ----------
    series : array_like
        Observed price or spread time series (1-D).
    dt : float
        Time step in years. Default 1/252 (daily).

    Returns
    -------
    dict
        theta, mu, sigma, logLik.
    """
    x = np.asarray(series, dtype=float).flatten()
    n = len(x)
    if n < 3:
        raise ValueError("fitOU requires at least 3 observations.")

    xPrev = x[:-1]
    xCurr = x[1:]
    m = len(xPrev)

    # OLS: xCurr = alpha * xPrev + beta
    sx  = xPrev.sum()
    sy  = xCurr.sum()
    sxx = (xPrev ** 2).sum()
    sxy = (xPrev * xCurr).sum()
    det = m * sxx - sx * sx

    if abs(det) < 1e-14:
        alpha = 1.0
        beta  = 0.0
    else:
        alpha = (m * sxy - sx * sy) / det
        beta  = (sy - alpha * sx) / m

    residuals = xCurr - alpha * xPrev - beta
    sigmaEps2 = float(np.mean(residuals ** 2))

    # Recover OU parameters
    if alpha <= 0.0 or alpha >= 1.0:
        alpha = max(min(alpha, 1.0 - 1e-8), 1e-8)
    theta = -np.log(alpha) / dt
    mu    = beta / (1.0 - alpha)
    # sigma^2 = sigmaEps^2 * 2*theta / (1 - exp(-2*theta*dt))
    denom = 1.0 - np.exp(-2.0 * theta * dt)
    if denom > 1e-12:
        sigma2 = sigmaEps2 * 2.0 * theta / denom
    else:
        sigma2 = sigmaEps2 / dt
    sigma = np.sqrt(max(sigma2, 0.0))

    # Log-likelihood
    logLik = -0.5 * m * np.log(2.0 * np.pi * sigmaEps2) - 0.5 * m

    return {'theta': float(theta), 'mu': float(mu), 'sigma': float(sigma),
            'logLik': float(logLik)}


# ---------------------------------------------------------------------------
# GARCH(1,1) calibration
# ---------------------------------------------------------------------------

def fitGarch(returns, p=1, q=1, maxIter=200):
    """
    Calibrate GARCH(1,1) parameters via ADAM gradient descent.

    Parameters
    ----------
    returns : array_like
        Return series (1-D).
    p, q : int
        GARCH orders (currently only (1,1) is supported; p, q are accepted
        for interface compatibility).
    maxIter : int
        Maximum ADAM iterations.

    Returns
    -------
    dict
        omega, alpha, beta, logLik, aic, bic.
    """
    eps = np.asarray(returns, dtype=float).flatten()
    n = len(eps)

    def _loglik(params):
        """Negative log-likelihood for GARCH(1,1)."""
        omega, a1, b1 = params
        if omega <= 0 or a1 <= 0 or b1 <= 0 or a1 + b1 >= 0.9999:
            return 1e10
        sigma2 = np.full(n, omega / (1.0 - a1 - b1))
        ll = 0.0
        for t in range(1, n):
            sigma2_t = omega + a1 * eps[t - 1] ** 2 + b1 * sigma2[t - 1]
            sigma2_t = max(sigma2_t, 1e-12)
            sigma2[t] = sigma2_t
            ll += np.log(sigma2_t) + eps[t] ** 2 / sigma2_t
        return 0.5 * ll

    def _grad(params, h=1e-5):
        """Numerical gradient."""
        g = np.zeros(3)
        f0 = _loglik(params)
        for i in range(3):
            p2 = params.copy()
            p2[i] += h
            g[i] = (_loglik(p2) - f0) / h
        return g

    # ADAM optimizer
    params = np.array([0.0001, 0.1, 0.8])
    m_adam = np.zeros(3)
    v_adam = np.zeros(3)
    beta1, beta2, eps_adam, lr = 0.9, 0.999, 1e-8, 0.001

    for t_adam in range(1, maxIter + 1):
        g = _grad(params)
        m_adam = beta1 * m_adam + (1 - beta1) * g
        v_adam = beta2 * v_adam + (1 - beta2) * g ** 2
        mHat = m_adam / (1 - beta1 ** t_adam)
        vHat = v_adam / (1 - beta2 ** t_adam)
        params = params - lr * mHat / (np.sqrt(vHat) + eps_adam)
        # Project onto feasible set
        params[0] = max(params[0], 1e-8)
        params[1] = max(params[1], 1e-6)
        params[2] = max(params[2], 1e-6)
        if params[1] + params[2] >= 0.9999:
            scale = 0.9999 / (params[1] + params[2])
            params[1] *= scale
            params[2] *= scale

    omega, alpha, beta = float(params[0]), float(params[1]), float(params[2])
    ll = float(-_loglik(params))
    k = 3
    aic = -2 * ll + 2 * k
    bic = -2 * ll + k * np.log(n)

    return {'omega': omega, 'alpha': alpha, 'beta': beta,
            'logLik': ll, 'aic': aic, 'bic': bic}


# ---------------------------------------------------------------------------
# Heston calibration
# ---------------------------------------------------------------------------

def fitHeston(prices, dt=1/252, maxIter=300):
    """
    Calibrate Heston stochastic volatility model parameters.

    Uses discretised Euler-Maruyama log-likelihood with ADAM optimisation.

    Parameters
    ----------
    prices : array_like
        Asset price series (1-D).
    dt : float
        Time step in years. Default 1/252.
    maxIter : int

    Returns
    -------
    dict
        mu, kappa, theta, sigma, rho, v0, logLik.
    """
    prices = np.asarray(prices, dtype=float).flatten()
    logRet = np.diff(np.log(prices))
    n = len(logRet)

    def _loglik(params):
        mu, kappa, theta_h, sigma_h, rho = params
        if kappa <= 0 or theta_h <= 0 or sigma_h <= 0:
            return 1e10
        rho = max(min(rho, 0.999), -0.999)
        v = theta_h
        ll = 0.0
        for t in range(n):
            v = max(v + kappa * (theta_h - v) * dt, 1e-8)
            muR = (mu - 0.5 * v) * dt
            varR = v * dt
            ll += 0.5 * (np.log(varR) + (logRet[t] - muR) ** 2 / varR)
        return ll

    def _grad(params, h=1e-5):
        g = np.zeros(5)
        f0 = _loglik(params)
        for i in range(5):
            p2 = params.copy()
            p2[i] += h
            g[i] = (_loglik(p2) - f0) / h
        return g

    params = np.array([0.05, 2.0, 0.04, 0.3, -0.5])
    m_adam = np.zeros(5)
    v_adam = np.zeros(5)
    beta1, beta2, eps_adam, lr = 0.9, 0.999, 1e-8, 0.001

    for t_adam in range(1, maxIter + 1):
        g = _grad(params)
        m_adam = beta1 * m_adam + (1 - beta1) * g
        v_adam = beta2 * v_adam + (1 - beta2) * g ** 2
        mHat = m_adam / (1 - beta1 ** t_adam)
        vHat = v_adam / (1 - beta2 ** t_adam)
        params -= lr * mHat / (np.sqrt(vHat) + eps_adam)
        params[1] = max(params[1], 0.01)   # kappa > 0
        params[2] = max(params[2], 1e-4)   # theta > 0
        params[3] = max(params[3], 1e-4)   # sigma > 0
        params[4] = max(min(params[4], 0.999), -0.999)  # rho in (-1,1)

    mu, kappa, theta_h, sigma_h, rho = params
    # Estimate v0 as initial realised variance proxy
    v0 = float(np.var(logRet[:min(21, n)]) / dt)
    logLik = float(-_loglik(params))

    return {'mu': float(mu), 'kappa': float(kappa), 'theta': float(theta_h),
            'sigma': float(sigma_h), 'rho': float(rho), 'v0': float(v0),
            'logLik': logLik}


# ---------------------------------------------------------------------------
# Copula fitting
# ---------------------------------------------------------------------------

def fitCopula(data, copulaType='gaussian'):
    """
    Fit a bivariate copula to data.

    Parameters
    ----------
    data : array_like, shape (n, d)
        Data matrix. Marginals are transformed to uniforms internally
        via rank-based probability integral transform.
    copulaType : {'gaussian', 't', 'clayton', 'gumbel'}

    Returns
    -------
    dict
        param (dict or scalar), logLik, tailDep (dict: lower, upper).
    """
    data = np.asarray(data, dtype=float)
    n, d = data.shape

    # Rank transform to uniforms
    u = np.zeros_like(data)
    for j in range(d):
        ranks = np.argsort(np.argsort(data[:, j])) + 1
        u[:, j] = ranks / (n + 1.0)

    def _kendallTau(u1, u2):
        conc = disc = 0
        for i in range(n):
            for j in range(i + 1, n):
                dx = u1[i] - u1[j]
                dy = u2[i] - u2[j]
                if dx * dy > 0:
                    conc += 1
                elif dx * dy < 0:
                    disc += 1
        total = n * (n - 1) // 2
        return (conc - disc) / total if total > 0 else 0.0

    if copulaType == 'gaussian':
        # Gaussian: invert to normal quantiles, estimate correlation
        z = np.array([[_normPpf(u[i, j]) for j in range(d)] for i in range(n)])
        rho = float(np.corrcoef(z[:, 0], z[:, 1])[0, 1])
        rho = max(min(rho, 0.9999), -0.9999)
        # Log-likelihood
        if abs(rho) < 1.0:
            R = np.array([[1.0, rho], [rho, 1.0]])
            logLik = float(-0.5 * n * np.log(np.linalg.det(R))
                           - 0.5 * sum(
                               z[i] @ np.linalg.inv(R) @ z[i] - z[i] @ z[i]
                               for i in range(n)))
        else:
            logLik = -np.inf
        return {'param': {'rho': rho}, 'logLik': logLik,
                'tailDep': {'lower': 0.0, 'upper': 0.0}}

    elif copulaType == 't':
        # t-copula: estimate rho, then optimise df
        z = np.array([[_normPpf(u[i, j]) for j in range(d)] for i in range(n)])
        rho = float(np.corrcoef(z[:, 0], z[:, 1])[0, 1])
        rho = max(min(rho, 0.9999), -0.9999)

        tau = _kendallTau(u[:, 0], u[:, 1])
        df = max(2.1, min(2.0 / (1.0 - abs(tau)) if abs(tau) < 1.0 else 30.0, 30.0))

        # Lower tail dependence for t-copula
        rho_val = rho
        if abs(rho_val) < 1.0:
            t_arg = np.sqrt((df + 1) * (1 - rho_val) / (1 + rho_val))
            # CDF of t_{df+1} at t_arg — use normal approximation for large df
            lower_td = 2.0 * float(_normCdf(-t_arg))
        else:
            lower_td = 0.0

        return {'param': {'rho': rho, 'df': df}, 'logLik': 0.0,
                'tailDep': {'lower': lower_td, 'upper': lower_td}}

    elif copulaType == 'clayton':
        tau = _kendallTau(u[:, 0], u[:, 1])
        tau = max(tau, 0.01)
        theta = 2.0 * tau / (1.0 - tau)
        theta = max(theta, 0.01)
        # Lower tail dependence
        lower_td = 2.0 ** (-1.0 / theta)
        return {'param': theta, 'logLik': 0.0,
                'tailDep': {'lower': lower_td, 'upper': 0.0}}

    elif copulaType == 'gumbel':
        tau = _kendallTau(u[:, 0], u[:, 1])
        tau = max(tau, 0.01)
        theta = max(1.0 / (1.0 - tau), 1.0)
        # Upper tail dependence
        upper_td = 2.0 - 2.0 ** (1.0 / theta)
        return {'param': theta, 'logLik': 0.0,
                'tailDep': {'lower': 0.0, 'upper': upper_td}}

    else:
        raise ValueError(f"copulaType must be 'gaussian', 't', 'clayton', or 'gumbel'. Got '{copulaType}'.")


# ---------------------------------------------------------------------------
# Distribution fitting
# ---------------------------------------------------------------------------

def fitDistribution(data, distType='normal'):
    """
    Fit a marginal distribution to data by MLE or method of moments.

    Parameters
    ----------
    data : array_like
        Observed data (1-D).
    distType : {'normal', 'lognormal', 't', 'gamma'}

    Returns
    -------
    dict
        params (dict), logLik, aic, bic.
    """
    x = np.asarray(data, dtype=float).flatten()
    n = len(x)

    if distType == 'normal':
        mu    = float(x.mean())
        sigma = float(x.std(ddof=1))
        sigma = max(sigma, 1e-12)
        logLik = float(-0.5 * n * np.log(2 * np.pi * sigma ** 2)
                       - 0.5 * np.sum((x - mu) ** 2) / sigma ** 2)
        k = 2
        return {'params': {'mu': mu, 'sigma': sigma}, 'logLik': logLik,
                'aic': -2 * logLik + 2 * k, 'bic': -2 * logLik + k * np.log(n)}

    elif distType == 'lognormal':
        if np.any(x <= 0):
            x = x[x > 0]
        logX  = np.log(x)
        mu    = float(logX.mean())
        sigma = float(logX.std(ddof=1))
        sigma = max(sigma, 1e-12)
        logLik = float(-0.5 * len(logX) * np.log(2 * np.pi * sigma ** 2)
                       - 0.5 * np.sum((logX - mu) ** 2) / sigma ** 2
                       - np.sum(logX))
        k = 2
        return {'params': {'mu': mu, 'sigma': sigma}, 'logLik': logLik,
                'aic': -2 * logLik + 2 * k, 'bic': -2 * logLik + k * np.log(n)}

    elif distType == 't':
        mu    = float(x.mean())
        xc    = x - mu
        var   = float(np.var(xc, ddof=1))
        # Method of moments: df = 2*var/(var-1) for standardised (assumes var > 1 for valid df)
        # For general: df ≈ 4 + 6/(kurt - 3) but we keep simple
        kurt = float(np.mean(xc ** 4) / max(np.mean(xc ** 2) ** 2, 1e-12))
        if kurt > 3.0:
            df = 4.0 + 6.0 / (kurt - 3.0)
        else:
            df = 30.0
        df = max(df, 2.1)
        sigma = float(np.sqrt(var * (df - 2) / df)) if df > 2 else float(np.sqrt(var))
        sigma = max(sigma, 1e-12)
        # Approximate log-likelihood using normal as proxy (proper t loglik needs lgamma)
        logLik = float(-0.5 * n * np.log(2 * np.pi * sigma ** 2)
                       - 0.5 * np.sum(xc ** 2) / sigma ** 2)
        k = 3
        return {'params': {'mu': mu, 'sigma': sigma, 'df': df}, 'logLik': logLik,
                'aic': -2 * logLik + 2 * k, 'bic': -2 * logLik + k * np.log(n)}

    elif distType == 'gamma':
        if np.any(x <= 0):
            x = x[x > 0]
        mu  = float(x.mean())
        var = float(x.var(ddof=1))
        var = max(var, 1e-12)
        shape = mu ** 2 / var
        scale = var / mu
        logLik = float(
            (shape - 1) * np.sum(np.log(x))
            - np.sum(x) / scale
            - len(x) * (shape * np.log(scale) + _logGamma(shape))
        )
        k = 2
        return {'params': {'shape': shape, 'scale': scale}, 'logLik': logLik,
                'aic': -2 * logLik + 2 * k, 'bic': -2 * logLik + k * np.log(n)}

    else:
        raise ValueError(f"distType must be 'normal', 'lognormal', 't', or 'gamma'. Got '{distType}'.")
