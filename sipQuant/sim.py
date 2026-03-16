import numpy as np


def gbm(mu, sigma, nSteps, nSims, s0=1.0, dt=1/252):
    """
    Geometric Brownian Motion.

    Parameters
    ----------
    mu : float
        Drift.
    sigma : float
        Volatility.
    nSteps : int
        Number of time steps.
    nSims : int
        Number of simulation paths.
    s0 : float, optional
        Initial price. Default 1.0.
    dt : float, optional
        Time increment. Default 1/252.

    Returns
    -------
    ndarray, shape (nSims, nSteps)
        Simulated price paths.
    """
    Z = np.random.randn(nSims, nSteps)
    inc = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    return s0 * np.exp(np.cumsum(inc, axis=1))


def ou(theta, mu, sigma, nSteps, nSims, x0=0.0, dt=1/252):
    """
    Ornstein-Uhlenbeck process.

    Parameters
    ----------
    theta : float
        Mean-reversion speed.
    mu : float
        Long-run mean.
    sigma : float
        Volatility.
    nSteps : int
        Number of time steps.
    nSims : int
        Number of simulation paths.
    x0 : float, optional
        Initial value. Default 0.0.
    dt : float, optional
        Time increment. Default 1/252.

    Returns
    -------
    ndarray, shape (nSims, nSteps)
        Simulated paths.
    """
    paths = np.zeros((nSims, nSteps))
    paths[:, 0] = x0
    Z = np.random.randn(nSims, nSteps)
    for t in range(1, nSteps):
        paths[:, t] = (paths[:, t-1]
                       + theta * (mu - paths[:, t-1]) * dt
                       + sigma * np.sqrt(dt) * Z[:, t])
    return paths


def levyOu(theta, mu, sigma, jumpLambda, jumpMu, jumpSigma,
           nSteps, nSims, x0=0.0, dt=1/252):
    """
    Levy-driven Ornstein-Uhlenbeck process with compound Poisson jumps.

    Parameters
    ----------
    theta : float
        Mean-reversion speed.
    mu : float
        Long-run mean.
    sigma : float
        Diffusion volatility.
    jumpLambda : float
        Poisson jump arrival rate.
    jumpMu : float
        Mean jump size.
    jumpSigma : float
        Std dev of jump size.
    nSteps : int
        Number of time steps.
    nSims : int
        Number of simulation paths.
    x0 : float, optional
        Initial value. Default 0.0.
    dt : float, optional
        Time increment. Default 1/252.

    Returns
    -------
    ndarray, shape (nSims, nSteps)
        Simulated paths.
    """
    paths = np.zeros((nSims, nSteps))
    paths[:, 0] = x0
    Z = np.random.randn(nSims, nSteps)
    nJumps = np.random.poisson(jumpLambda * dt, (nSims, nSteps))
    jumpSizes = np.random.normal(jumpMu, jumpSigma, (nSims, nSteps)) * (nJumps > 0)
    for t in range(1, nSteps):
        paths[:, t] = (paths[:, t-1]
                       + theta * (mu - paths[:, t-1]) * dt
                       + sigma * np.sqrt(dt) * Z[:, t]
                       + jumpSizes[:, t])
    return paths


def arma(arCoefs, maCoefs, sigma, nSteps, nSims, x0=0.0):
    """
    ARMA(p, q) simulation.

    Parameters
    ----------
    arCoefs : array_like
        AR coefficients (length p).
    maCoefs : array_like
        MA coefficients (length q).
    sigma : float
        Innovation standard deviation.
    nSteps : int
        Number of time steps.
    nSims : int
        Number of simulation paths.
    x0 : float, optional
        Initial value. Default 0.0.

    Returns
    -------
    ndarray, shape (nSims, nSteps)
        Simulated paths.
    """
    p, q = len(arCoefs), len(maCoefs)
    arCoefs = np.asarray(arCoefs)
    maCoefs = np.asarray(maCoefs)
    paths = np.zeros((nSims, nSteps))
    eps = np.random.normal(0, sigma, (nSims, nSteps))
    paths[:, 0] = x0
    for t in range(1, nSteps):
        ar = sum(arCoefs[j] * paths[:, t-j-1] for j in range(min(p, t)))
        ma = sum(maCoefs[j] * eps[:, t-j-1] for j in range(min(q, t)))
        paths[:, t] = ar + ma + eps[:, t]
    return paths


def markovSwitching(mu1, sigma1, mu2, sigma2, p11, p22, nSteps, nSims, x0=0.0):
    """
    Two-state Markov switching model.

    Parameters
    ----------
    mu1 : float
        Drift in state 0.
    sigma1 : float
        Volatility in state 0.
    mu2 : float
        Drift in state 1.
    sigma2 : float
        Volatility in state 1.
    p11 : float
        Probability of staying in state 0.
    p22 : float
        Probability of staying in state 1.
    nSteps : int
        Number of time steps.
    nSims : int
        Number of simulation paths.
    x0 : float, optional
        Initial value. Default 0.0.

    Returns
    -------
    ndarray, shape (nSims, nSteps)
        Simulated paths.
    """
    paths = np.zeros((nSims, nSteps))
    paths[:, 0] = x0
    states = np.zeros(nSims, dtype=int)
    for t in range(1, nSteps):
        stay = np.random.rand(nSims)
        new_states = states.copy()
        in0 = states == 0
        in1 = states == 1
        new_states[in0] = np.where(stay[in0] < p11, 0, 1)
        new_states[in1] = np.where(stay[in1] < p22, 1, 0)
        mu = np.where(new_states == 0, mu1, mu2)
        sig = np.where(new_states == 0, sigma1, sigma2)
        paths[:, t] = paths[:, t-1] + np.random.normal(mu, sig)
        states = new_states
    return paths


def garch(omega, alpha1, beta1, nSteps, nSims):
    """
    GARCH(1, 1) simulation.

    Parameters
    ----------
    omega : float
        Constant term in variance equation.
    alpha1 : float
        ARCH coefficient.
    beta1 : float
        GARCH coefficient.
    nSteps : int
        Number of time steps.
    nSims : int
        Number of simulation paths.

    Returns
    -------
    ndarray, shape (nSims, nSteps)
        Simulated return paths.
    """
    paths = np.zeros((nSims, nSteps))
    sigma2 = np.full(nSims, omega / max(1 - alpha1 - beta1, 1e-8))
    Z = np.random.randn(nSims, nSteps)
    for t in range(nSteps):
        paths[:, t] = Z[:, t] * np.sqrt(sigma2)
        if t < nSteps - 1:
            sigma2 = omega + alpha1 * paths[:, t]**2 + beta1 * sigma2
    return paths


def heston(mu, kappa, theta, sigma, rho, s0=100.0, v0=0.04,
           nSteps=252, nSims=1000, dt=1/252):
    """
    Heston stochastic volatility model.

    Parameters
    ----------
    mu : float
        Drift of the underlying asset.
    kappa : float
        Mean-reversion speed of variance.
    theta : float
        Long-run variance.
    sigma : float
        Volatility of variance (vol-of-vol).
    rho : float
        Correlation between asset and variance Brownian motions.
    s0 : float, optional
        Initial asset price. Default 100.0.
    v0 : float, optional
        Initial variance. Default 0.04.
    nSteps : int, optional
        Number of time steps. Default 252.
    nSims : int, optional
        Number of simulation paths. Default 1000.
    dt : float, optional
        Time increment. Default 1/252.

    Returns
    -------
    dict
        prices : ndarray, shape (nSims, nSteps)
        variances : ndarray, shape (nSims, nSteps)
    """
    prices = np.zeros((nSims, nSteps))
    variances = np.zeros((nSims, nSteps))
    prices[:, 0] = s0
    variances[:, 0] = v0
    Z1 = np.random.randn(nSims, nSteps)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(nSims, nSteps)
    for t in range(1, nSteps):
        vt = np.maximum(variances[:, t-1], 0)
        variances[:, t] = np.maximum(
            vt + kappa * (theta - vt) * dt + sigma * np.sqrt(vt * dt) * Z2[:, t], 0)
        prices[:, t] = prices[:, t-1] * np.exp(
            (mu - 0.5 * vt) * dt + np.sqrt(vt * dt) * Z1[:, t])
    return {'prices': prices, 'variances': variances}


def compoundPoisson(lambdaRate, jumpMu, jumpSigma, nSteps, nSims, s0=100.0, dt=1/252):
    """
    Compound Poisson process.

    Parameters
    ----------
    lambdaRate : float
        Poisson arrival rate per unit time.
    jumpMu : float
        Mean jump size.
    jumpSigma : float
        Std dev of jump size.
    nSteps : int
        Number of time steps.
    nSims : int
        Number of simulation paths.
    s0 : float, optional
        Initial level. Default 100.0.
    dt : float, optional
        Time increment. Default 1/252.

    Returns
    -------
    ndarray, shape (nSims, nSteps)
        Simulated paths.
    """
    paths = np.zeros((nSims, nSteps))
    paths[:, 0] = s0
    nJumps = np.random.poisson(lambdaRate * dt, (nSims, nSteps))
    jumpAmts = np.random.normal(jumpMu, jumpSigma, (nSims, nSteps)) * nJumps
    for t in range(1, nSteps):
        paths[:, t] = paths[:, t-1] + jumpAmts[:, t]
    return paths


def simulate(model, params, nSteps, nSims):
    """
    Dispatcher for all simulation models.

    Parameters
    ----------
    model : str
        Model name. Supported: 'gbm', 'ou', 'levyou', 'arma',
        'markovswitching', 'garch', 'heston', 'compoundpoisson'.
    params : dict
        Keyword arguments forwarded to the selected model function
        (excluding *nSteps* and *nSims*).
    nSteps : int
        Number of time steps.
    nSims : int
        Number of simulation paths.

    Returns
    -------
    ndarray or dict
        Output of the selected model function.

    Raises
    ------
    ValueError
        If *model* is not recognised.
    """
    m = model.lower().replace('_', '').replace('-', '')
    dispatch = {
        'gbm': gbm,
        'ou': ou,
        'levyou': levyOu,
        'arma': arma,
        'markovswitching': markovSwitching,
        'garch': garch,
        'heston': heston,
        'compoundpoisson': compoundPoisson,
    }
    if m not in dispatch:
        raise ValueError(f"Unknown model '{model}'. Available: {list(dispatch.keys())}")
    return dispatch[m](nSteps=nSteps, nSims=nSims, **params)
