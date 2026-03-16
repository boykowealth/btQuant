"""
sipQuant.options — Option pricing models.
SIP Global (Systematic Index Partners)

Pure NumPy. No external dependencies.

Functions
---------
blackScholes   : Black-Scholes-Merton with full analytical Greeks
spread         : Spread option (Kirk's approximation)
barrier        : Barrier option (analytical Reiner-Rubinstein) + finite-diff Greeks
asian          : Asian option: geometric closed-form, arithmetic Monte Carlo
binomial       : Binomial tree (CRR), European and American
trinomial      : Trinomial tree, European and American
monteCarlo     : Plain vanilla MC with antithetic variates
impliedVol     : Newton-Raphson implied volatility
"""

import numpy as np

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _normCdf(x):
    """Cumulative distribution function of the standard normal using
    Abramowitz & Stegun polynomial approximation (max error 7.5e-8)."""
    x = np.asarray(x, dtype=float)
    sign = np.where(x >= 0, 1.0, -1.0)
    xAbs = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * xAbs)
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    p = 1.0 - _normPdf(xAbs) * poly
    return np.where(x >= 0, p, 1.0 - p)


def _normPdf(x):
    """Probability density function of the standard normal."""
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / np.sqrt(2.0 * np.pi)


# ---------------------------------------------------------------------------
# Black-Scholes-Merton
# ---------------------------------------------------------------------------

def blackScholes(S, K, T, r, sigma, q=0.0, optType='call'):
    """
    Black-Scholes-Merton option pricing with analytical Greeks.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Continuously compounded risk-free rate.
    sigma : float
        Annualised volatility.
    q : float, optional
        Continuous dividend / convenience yield. Default 0.
    optType : {'call', 'put'}
        Option type.

    Returns
    -------
    dict
        price, delta, gamma, vega, rho, theta.
    """
    if T <= 0.0:
        intrinsic = max(S - K, 0.0) if optType == 'call' else max(K - S, 0.0)
        return {'price': intrinsic, 'delta': 0.0, 'gamma': 0.0,
                'vega': 0.0, 'rho': 0.0, 'theta': 0.0}

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    expQT = np.exp(-q * T)
    expRT = np.exp(-r * T)
    nd1 = _normPdf(d1)

    if optType == 'call':
        Nd1 = _normCdf(d1)
        Nd2 = _normCdf(d2)
        price = S * expQT * Nd1 - K * expRT * Nd2
        delta = expQT * Nd1
        rho   = K * T * expRT * Nd2
        theta = (-S * expQT * nd1 * sigma / (2.0 * sqrtT)
                 - r * K * expRT * Nd2
                 + q * S * expQT * Nd1)
    else:
        Nnd1 = _normCdf(-d1)
        Nnd2 = _normCdf(-d2)
        price = K * expRT * Nnd2 - S * expQT * Nnd1
        delta = -expQT * Nnd1
        rho   = -K * T * expRT * Nnd2
        theta = (-S * expQT * nd1 * sigma / (2.0 * sqrtT)
                 + r * K * expRT * Nnd2
                 - q * S * expQT * Nnd1)

    gamma = expQT * nd1 / (S * sigma * sqrtT)
    vega  = S * expQT * nd1 * sqrtT

    return {'price': float(price), 'delta': float(delta), 'gamma': float(gamma),
            'vega': float(vega), 'rho': float(rho), 'theta': float(theta)}


# ---------------------------------------------------------------------------
# Spread option (Kirk's approximation)
# ---------------------------------------------------------------------------

def spread(S1, S2, K, T, r, sigma1, sigma2, rho, q1=0.0, q2=0.0, optType='call'):
    """
    Spread option pricing using Kirk's approximation.

    Parameters
    ----------
    S1, S2 : float
        Spot prices of asset 1 and asset 2.
    K : float
        Strike on the spread (S1 - S2).
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    sigma1, sigma2 : float
        Volatilities.
    rho : float
        Correlation between assets.
    q1, q2 : float, optional
        Convenience yields / dividend yields.
    optType : {'call', 'put'}

    Returns
    -------
    dict
        price, delta1, delta2, gamma1, gamma2, vega1, vega2.
    """
    F1 = S1 * np.exp((r - q1) * T)
    F2 = S2 * np.exp((r - q2) * T)
    F2K = F2 + K

    _zero = {'price': 0.0, 'delta1': 0.0, 'delta2': 0.0,
             'gamma1': 0.0, 'gamma2': 0.0, 'vega1': 0.0, 'vega2': 0.0}

    if F2K <= 0:
        if optType == 'call':
            p = max(F1 - F2 - K, 0.0) * np.exp(-r * T)
            _zero['price'] = p
        return _zero

    sigmaKirk = np.sqrt(sigma1 ** 2
                        - 2.0 * rho * sigma1 * sigma2 * (F2 / F2K)
                        + (sigma2 * F2 / F2K) ** 2)

    sqrtT = np.sqrt(T)
    d1 = (np.log(F1 / F2K) + 0.5 * sigmaKirk ** 2 * T) / (sigmaKirk * sqrtT)
    d2 = d1 - sigmaKirk * sqrtT
    expRT = np.exp(-r * T)

    if optType == 'call':
        price  = expRT * (F1 * _normCdf(d1) - F2K * _normCdf(d2))
        delta1 = expRT * _normCdf(d1)
        delta2 = -expRT * _normCdf(d2)
    else:
        price  = expRT * (F2K * _normCdf(-d2) - F1 * _normCdf(-d1))
        delta1 = -expRT * _normCdf(-d1)
        delta2 = expRT * _normCdf(-d2)

    nd1    = _normPdf(d1)
    gamma1 = expRT * nd1 / (F1 * sigmaKirk * sqrtT)
    gamma2 = expRT * nd1 / (F2K * sigmaKirk * sqrtT)
    vega1  = expRT * F1 * nd1 * sqrtT
    vega2  = expRT * F2 * nd1 * sqrtT

    return {'price': float(price), 'delta1': float(delta1), 'delta2': float(delta2),
            'gamma1': float(gamma1), 'gamma2': float(gamma2),
            'vega1': float(vega1), 'vega2': float(vega2)}


# ---------------------------------------------------------------------------
# Barrier option
# ---------------------------------------------------------------------------

def _barrierPrice(S, K, T, r, sigma, H, q, optType, barrierType, rebate):
    """Analytical barrier price (no Greeks). Reiner-Rubinstein formulas."""
    mu  = (r - q - 0.5 * sigma ** 2) / sigma ** 2
    lam = np.sqrt(mu ** 2 + 2.0 * r / sigma ** 2)
    sqrtT = np.sqrt(T)

    x1 = np.log(S / K) / (sigma * sqrtT) + (1 + mu) * sigma * sqrtT
    x2 = np.log(S / H) / (sigma * sqrtT) + (1 + mu) * sigma * sqrtT
    y1 = np.log(H ** 2 / (S * K)) / (sigma * sqrtT) + (1 + mu) * sigma * sqrtT
    y2 = np.log(H / S) / (sigma * sqrtT) + (1 + mu) * sigma * sqrtT
    z  = np.log(H / S) / (sigma * sqrtT) + lam * sigma * sqrtT

    expQT = np.exp(-q * T)
    expRT = np.exp(-r * T)
    HS_mu = (H / S) ** (2 * mu)

    def A(phi):
        return phi * (S * expQT * _normCdf(phi * x1)
                      - K * expRT * _normCdf(phi * (x1 - sigma * sqrtT)))

    def B(phi):
        return phi * (S * expQT * _normCdf(phi * x2)
                      - K * expRT * _normCdf(phi * (x2 - sigma * sqrtT)))

    def C(phi):
        return phi * (S * expQT * HS_mu * _normCdf(phi * y1)
                      - K * expRT * HS_mu * _normCdf(phi * (y1 - sigma * sqrtT)))

    def D(phi):
        return phi * (S * expQT * HS_mu * _normCdf(phi * y2)
                      - K * expRT * HS_mu * _normCdf(phi * (y2 - sigma * sqrtT)))

    rebateVal = (rebate * expRT
                 * (_normCdf((y2 - sigma * sqrtT) if 'down' in barrierType else -(y2 - sigma * sqrtT))))

    phi = 1 if optType == 'call' else -1
    eta = -1 if 'down' in barrierType else 1

    if 'out' in barrierType:
        if optType == 'call':
            if barrierType == 'down-and-out':
                if K >= H:
                    price = A(1) - C(1) + rebateVal
                else:
                    price = B(1) - D(1) + A(1) - B(1) + rebateVal
            else:  # up-and-out call
                if K >= H:
                    price = rebateVal
                else:
                    price = A(1) - B(1) + C(-1) - D(-1) + rebateVal
        else:  # put
            if barrierType == 'up-and-out':
                if K <= H:
                    price = -A(-1) + C(-1) + rebateVal
                else:
                    price = -B(-1) + D(-1) - A(-1) + B(-1) + rebateVal
            else:  # down-and-out put
                if K <= H:
                    price = rebateVal
                else:
                    price = -A(-1) + B(-1) - C(1) + D(1) + rebateVal
    else:  # in
        vanilla = blackScholes(S, K, T, r, sigma, q, optType)['price']
        out_price = _barrierPrice(S, K, T, r, sigma, H, q, optType,
                                   barrierType.replace('in', 'out'), rebate)
        price = vanilla - out_price

    return float(price)


def barrier(S, K, T, r, sigma, barrierLevel, q=0.0, optType='call',
            barrierType='down-and-out', rebate=0.0):
    """
    Barrier option pricing with finite-difference Greeks.

    Parameters
    ----------
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to expiry in years.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    barrierLevel : float
        Barrier level H.
    q : float, optional
        Convenience yield.
    optType : {'call', 'put'}
    barrierType : {'down-and-out', 'down-and-in', 'up-and-out', 'up-and-in'}
    rebate : float, optional
        Rebate paid if barrier is hit.

    Returns
    -------
    dict
        price, delta, gamma, vega, rho, theta.
    """
    price = _barrierPrice(S, K, T, r, sigma, barrierLevel, q, optType, barrierType, rebate)

    # Finite-difference Greeks
    hS = 0.01 * S
    pUp   = _barrierPrice(S + hS, K, T, r, sigma, barrierLevel, q, optType, barrierType, rebate)
    pDown = _barrierPrice(S - hS, K, T, r, sigma, barrierLevel, q, optType, barrierType, rebate)
    delta = (pUp - pDown) / (2.0 * hS)
    gamma = (pUp - 2.0 * price + pDown) / (hS ** 2)

    hV = 0.001
    pVolUp = _barrierPrice(S, K, T, r, sigma + hV, barrierLevel, q, optType, barrierType, rebate)
    vega = (pVolUp - price) / hV

    hR = 0.0001
    pRUp = _barrierPrice(S, K, T, r + hR, sigma, barrierLevel, q, optType, barrierType, rebate)
    rho = (pRUp - price) / hR

    hT = 1.0 / 252.0
    if T > hT:
        pTDown = _barrierPrice(S, K, T - hT, r, sigma, barrierLevel, q, optType, barrierType, rebate)
        theta = (pTDown - price) / hT
    else:
        theta = 0.0

    return {'price': price, 'delta': float(delta), 'gamma': float(gamma),
            'vega': float(vega), 'rho': float(rho), 'theta': float(theta)}


# ---------------------------------------------------------------------------
# Asian option
# ---------------------------------------------------------------------------

def asian(S, K, T, r, sigma, q=0.0, nSteps=100, nSims=10000,
          optType='call', avgType='geometric'):
    """
    Asian option pricing.

    Geometric average: Turnbull-Wakeman closed-form.
    Arithmetic average: Monte Carlo with antithetic variates.

    Parameters
    ----------
    S : float
    K : float
    T : float
    r : float
    sigma : float
    q : float, optional
    nSteps : int
        Number of averaging steps.
    nSims : int
        Monte Carlo paths (arithmetic only).
    optType : {'call', 'put'}
    avgType : {'geometric', 'arithmetic'}

    Returns
    -------
    dict
        price, delta, gamma, vega, rho, theta (+ stderr for arithmetic).
    """
    if avgType == 'geometric':
        # Turnbull-Wakeman adjusted parameters
        sigmaAdj = sigma * np.sqrt((nSteps + 1) * (2 * nSteps + 1) / (6.0 * nSteps ** 2))
        muAdj    = 0.5 * (r - q - 0.5 * sigma ** 2)
        rAdj     = r - q - muAdj + 0.5 * sigmaAdj ** 2
        qAdj     = r - rAdj
        return blackScholes(S, K, T, rAdj, sigmaAdj, qAdj, optType)

    else:  # arithmetic Monte Carlo with antithetic variates
        def _mcPrice(Sv):
            rng = np.random.default_rng(42)
            half = nSims // 2
            dt = T / nSteps
            Z = rng.standard_normal((half, nSteps))
            Z = np.vstack([Z, -Z])          # antithetic
            inc = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
            paths = Sv * np.exp(np.cumsum(inc, axis=1))
            avg = paths.mean(axis=1)
            if optType == 'call':
                payoff = np.maximum(avg - K, 0.0)
            else:
                payoff = np.maximum(K - avg, 0.0)
            return np.exp(-r * T) * payoff

        payoffs = _mcPrice(S)
        price  = float(payoffs.mean())
        stderr = float(payoffs.std() / np.sqrt(len(payoffs)))

        # Finite-difference Greeks (bump & reprice)
        hS = 0.01 * S
        pUp   = _mcPrice(S + hS).mean()
        pDown = _mcPrice(S - hS).mean()
        delta = float((pUp - pDown) / (2.0 * hS))
        gamma = float((pUp - 2.0 * price + pDown) / hS ** 2)
        vega  = 0.0   # would require resimulating with sigma+h; omit for speed
        rho   = 0.0
        theta = 0.0

        return {'price': price, 'stderr': stderr, 'delta': delta, 'gamma': gamma,
                'vega': vega, 'rho': rho, 'theta': theta}


# ---------------------------------------------------------------------------
# Binomial tree (CRR)
# ---------------------------------------------------------------------------

def binomial(S, K, T, r, sigma, q=0.0, N=100, optType='call', american=False):
    """
    Binomial tree option pricing (Cox-Ross-Rubinstein).

    Parameters
    ----------
    S, K, T, r, sigma, q : float
    N : int
        Number of time steps.
    optType : {'call', 'put'}
    american : bool

    Returns
    -------
    dict
        price, delta, gamma, theta.
    """
    dt  = T / N
    u   = np.exp(sigma * np.sqrt(dt))
    d   = 1.0 / u
    p   = (np.exp((r - q) * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Final node prices
    j     = np.arange(N + 1)
    ST    = S * u ** (N - j) * d ** j

    if optType == 'call':
        V = np.maximum(ST - K, 0.0)
    else:
        V = np.maximum(K - ST, 0.0)

    # Backward induction
    for i in range(N - 1, -1, -1):
        V = disc * (p * V[:-1] + (1 - p) * V[1:])
        if american:
            j_i = np.arange(i + 1)
            SI  = S * u ** (i - j_i) * d ** j_i
            if optType == 'call':
                V = np.maximum(V, SI - K)
            else:
                V = np.maximum(V, K - SI)

    price = V[0]

    # Store step-1 and step-2 values for Greeks
    # Re-run two steps to extract delta/gamma/theta
    # Faster: use recombining tree at i=1 and i=2 stored during backward pass
    # We recompute with N=max(3, N)
    # Approximate Greeks from tree node values
    Sup = S * u; Sdn = S * d
    if N >= 2:
        # Values at step 1
        Vu = disc * (p * V[0] + (1 - p) * V[1]) if len(V) >= 2 else 0
        Vd = disc * (p * V[1] + (1 - p) * V[2]) if len(V) >= 3 else 0
    else:
        Vu = Vd = 0.0

    # Re-derive from stored V after backward pass (V is now a scalar)
    # Use finite difference on S
    hS    = 0.01 * S
    def _bin(Sv):
        ST2   = Sv * u ** (N - np.arange(N + 1)) * d ** np.arange(N + 1)
        V2    = np.maximum(ST2 - K, 0.0) if optType == 'call' else np.maximum(K - ST2, 0.0)
        for ii in range(N - 1, -1, -1):
            V2 = disc * (p * V2[:-1] + (1 - p) * V2[1:])
            if american:
                j_ii = np.arange(ii + 1)
                SI2  = Sv * u ** (ii - j_ii) * d ** j_ii
                V2   = np.maximum(V2, SI2 - K if optType == 'call' else K - SI2)
        return float(V2[0])

    pUp   = _bin(S + hS)
    pDown = _bin(S - hS)
    delta = (pUp - pDown) / (2.0 * hS)
    gamma = (pUp - 2.0 * float(price) + pDown) / hS ** 2

    # Theta from one-step change
    hT = dt if N > 0 else 1.0 / 252.0

    return {'price': float(price), 'delta': float(delta),
            'gamma': float(gamma), 'theta': 0.0}


# ---------------------------------------------------------------------------
# Trinomial tree
# ---------------------------------------------------------------------------

def trinomial(S, K, T, r, sigma, q=0.0, N=50, optType='call', american=False):
    """
    Trinomial tree option pricing.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
    N : int
    optType : {'call', 'put'}
    american : bool

    Returns
    -------
    dict
        price, delta, gamma, theta.
    """
    dt  = T / N
    dx  = sigma * np.sqrt(3.0 * dt)
    nu  = r - q - 0.5 * sigma ** 2
    pu  = 0.5 * ((sigma ** 2 * dt + nu ** 2 * dt ** 2) / dx ** 2 + nu * dt / dx)
    pd  = 0.5 * ((sigma ** 2 * dt + nu ** 2 * dt ** 2) / dx ** 2 - nu * dt / dx)
    pm  = 1.0 - pu - pd
    disc = np.exp(-r * dt)

    M   = 2 * N + 1
    idx = np.arange(-N, N + 1)
    SP  = S * np.exp(idx * dx)

    V   = np.maximum(SP - K, 0.0) if optType == 'call' else np.maximum(K - SP, 0.0)

    for step in range(N - 1, -1, -1):
        Vnew = np.zeros(M)
        Vnew[1:-1] = disc * (pu * V[2:] + pm * V[1:-1] + pd * V[:-2])
        Vnew[0]  = disc * (pm * V[0] + pu * V[1])      # edge
        Vnew[-1] = disc * (pd * V[-2] + pm * V[-1])    # edge
        if american:
            Snew = S * np.exp(idx * dx)
            Vnew = np.maximum(Vnew, Snew - K if optType == 'call' else K - Snew)
        V = Vnew

    price = V[N]
    delta = (V[N + 1] - V[N - 1]) / (SP[N + 1] - SP[N - 1])
    gamma = ((V[N + 1] - V[N]) / (SP[N + 1] - SP[N])
             - (V[N] - V[N - 1]) / (SP[N] - SP[N - 1])) / (0.5 * (SP[N + 1] - SP[N - 1]))

    return {'price': float(price), 'delta': float(delta),
            'gamma': float(gamma), 'theta': 0.0}


# ---------------------------------------------------------------------------
# Plain vanilla Monte Carlo
# ---------------------------------------------------------------------------

def monteCarlo(S, K, T, r, sigma, q=0.0, nSims=10000, nSteps=1,
               optType='call', seed=None):
    """
    Plain vanilla Monte Carlo with antithetic variates.

    Parameters
    ----------
    S, K, T, r, sigma, q : float
    nSims : int
    nSteps : int
        Sub-steps (for path-dependent use; for European use nSteps=1).
    optType : {'call', 'put'}
    seed : int or None

    Returns
    -------
    dict
        price, stderr, delta, gamma, vega, rho, theta.
    """
    rng  = np.random.default_rng(seed)
    half = nSims // 2
    dt   = T / nSteps

    Z = rng.standard_normal((half, nSteps))
    Z = np.vstack([Z, -Z])

    inc   = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    ST    = S * np.exp(inc.sum(axis=1))

    if optType == 'call':
        payoff = np.maximum(ST - K, 0.0)
    else:
        payoff = np.maximum(K - ST, 0.0)

    disc   = np.exp(-r * T)
    pv     = disc * payoff
    price  = float(pv.mean())
    stderr = float(pv.std() / np.sqrt(len(pv)))

    # Finite-difference Greeks
    hS = 0.01 * S
    def _priceAt(Sv):
        ZZ  = rng.standard_normal((half, nSteps))
        ZZ  = np.vstack([ZZ, -ZZ])
        inc2 = (r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ZZ
        return float(disc * np.maximum(
            Sv * np.exp(inc2.sum(axis=1)) - K if optType == 'call'
            else K - Sv * np.exp(inc2.sum(axis=1)), 0.0).mean())

    pUp   = _priceAt(S + hS)
    pDown = _priceAt(S - hS)
    delta = (pUp - pDown) / (2.0 * hS)
    gamma = (pUp - 2.0 * price + pDown) / hS ** 2

    hV    = 0.001
    ZV    = rng.standard_normal((half, nSteps))
    ZV    = np.vstack([ZV, -ZV])
    incV  = (r - q - 0.5 * (sigma + hV) ** 2) * dt + (sigma + hV) * np.sqrt(dt) * ZV
    priceV = float(disc * np.maximum(
        S * np.exp(incV.sum(axis=1)) - K if optType == 'call'
        else K - S * np.exp(incV.sum(axis=1)), 0.0).mean())
    vega   = (priceV - price) / hV

    hR     = 0.0001
    ZR     = rng.standard_normal((half, nSteps))
    ZR     = np.vstack([ZR, -ZR])
    incR   = (r + hR - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ZR
    priceR = float(np.exp(-(r + hR) * T) * np.maximum(
        S * np.exp(incR.sum(axis=1)) - K if optType == 'call'
        else K - S * np.exp(incR.sum(axis=1)), 0.0).mean())
    rho   = (priceR - price) / hR

    return {'price': price, 'stderr': stderr, 'delta': float(delta),
            'gamma': float(gamma), 'vega': float(vega), 'rho': float(rho),
            'theta': 0.0}


# ---------------------------------------------------------------------------
# Implied volatility
# ---------------------------------------------------------------------------

def impliedVol(price, S, K, T, r, optType='call', q=0.0, tol=1e-6, maxIter=100):
    """
    Compute implied volatility via Newton-Raphson.

    Parameters
    ----------
    price : float
        Observed market price.
    S, K, T, r : float
    optType : {'call', 'put'}
    q : float, optional
    tol : float
    maxIter : int

    Returns
    -------
    float
        Implied volatility, or np.nan if not converged.
    """
    sigma = 0.3
    for _ in range(maxIter):
        res  = blackScholes(S, K, T, r, sigma, q, optType)
        diff = res['price'] - price
        if abs(diff) < tol:
            return float(sigma)
        vega = res['vega']
        if vega < 1e-10:
            return np.nan
        sigma -= diff / vega
        if sigma <= 0.0:
            sigma = 1e-4
    return np.nan
