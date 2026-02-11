import numpy as np

def gbm(mu, sigma, nSteps, nSims, s0=1.0, dt=1/252):
    """
    Geometric Brownian Motion simulation.
    
    Parameters:
        mu: drift
        sigma: volatility
        nSteps: number of time steps
        nSims: number of simulations
        s0: initial value
        dt: time step
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        shocks = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), nSteps)
        sims[i] = s0 * np.exp(np.cumsum(shocks))
    
    return sims

def ou(theta, mu, sigma, nSteps, nSims, x0=0.0, dt=1/252):
    """
    Ornstein-Uhlenbeck process simulation.
    
    Parameters:
        theta: mean reversion rate
        mu: long-term mean
        sigma: volatility
        nSteps: number of time steps
        nSims: number of simulations
        x0: initial value
        dt: time step
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        x = np.zeros(nSteps)
        x[0] = x0
        
        for t in range(1, nSteps):
            dx = theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
            x[t] = x[t - 1] + dx
        
        sims[i] = x
    
    return sims

def levyOu(theta, mu, sigma, jumpLambda, jumpMu, jumpSigma, nSteps, nSims, x0=0.0, dt=1/252):
    """
    Lévy OU process (OU with jumps) simulation.
    
    Parameters:
        theta: mean reversion rate
        mu: long-term mean
        sigma: diffusion volatility
        jumpLambda: jump intensity
        jumpMu: jump mean
        jumpSigma: jump volatility
        nSteps: number of time steps
        nSims: number of simulations
        x0: initial value
        dt: time step
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        x = np.zeros(nSteps)
        x[0] = x0
        
        for t in range(1, nSteps):
            jump = 0
            if np.random.rand() < jumpLambda * dt:
                jump = np.random.normal(jumpMu, jumpSigma)
            
            dx = theta * (mu - x[t - 1]) * dt + sigma * np.sqrt(dt) * np.random.randn() + jump
            x[t] = x[t - 1] + dx
        
        sims[i] = x
    
    return sims

def ar1(phi, intercept, sigma, nSteps, nSims, x0=0.0):
    """
    AR(1) process simulation.
    
    Parameters:
        phi: autoregressive coefficient
        intercept: constant term
        sigma: error variance
        nSteps: number of time steps
        nSims: number of simulations
        x0: initial value
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        x = np.zeros(nSteps)
        x[0] = x0
        
        for t in range(1, nSteps):
            x[t] = intercept + phi * x[t - 1] + np.random.normal(0, np.sqrt(sigma))
        
        sims[i] = x
    
    return sims

def arma(arCoefs, maCoefs, sigma, nSteps, nSims, x0=0.0):
    """
    ARMA process simulation.
    
    Parameters:
        arCoefs: AR coefficients (list)
        maCoefs: MA coefficients (list)
        sigma: error std deviation
        nSteps: number of time steps
        nSims: number of simulations
        x0: initial value
    
    Returns:
        array (nSims x nSteps)
    """
    p = len(arCoefs)
    q = len(maCoefs)
    
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        x = np.zeros(nSteps)
        errors = np.random.normal(0, sigma, nSteps)
        
        x[0] = x0
        
        for t in range(1, nSteps):
            arTerm = sum(arCoefs[j] * x[t - j - 1] for j in range(min(p, t)))
            maTerm = sum(maCoefs[j] * errors[t - j - 1] for j in range(min(q, t)))
            x[t] = arTerm + maTerm + errors[t]
        
        sims[i] = x
    
    return sims

def markovSwitching(mu1, sigma1, mu2, sigma2, p11, p22, nSteps, nSims, x0=0.0):
    """
    Markov regime switching model simulation.
    
    Parameters:
        mu1: mean in regime 1
        sigma1: std dev in regime 1
        mu2: mean in regime 2
        sigma2: std dev in regime 2
        p11: probability of staying in regime 1
        p22: probability of staying in regime 2
        nSteps: number of time steps
        nSims: number of simulations
        x0: initial value
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        x = np.zeros(nSteps)
        x[0] = x0
        state = 0
        
        for t in range(1, nSteps):
            if state == 0:
                x[t] = x[t - 1] + np.random.normal(mu1, sigma1)
                state = 0 if np.random.rand() < p11 else 1
            else:
                x[t] = x[t - 1] + np.random.normal(mu2, sigma2)
                state = 1 if np.random.rand() < p22 else 0
        
        sims[i] = x
    
    return sims

def arch(alpha0, alpha1, nSteps, nSims):
    """
    ARCH(1) process simulation.
    
    Parameters:
        alpha0: constant term
        alpha1: ARCH coefficient
        nSteps: number of time steps
        nSims: number of simulations
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        x = np.zeros(nSteps)
        sigma2 = np.zeros(nSteps)
        sigma2[0] = alpha0 / (1 - alpha1) if alpha1 < 1 else 1.0
        
        for t in range(nSteps):
            eps = np.random.randn()
            x[t] = eps * np.sqrt(sigma2[t])
            if t < nSteps - 1:
                sigma2[t + 1] = alpha0 + alpha1 * x[t]**2
        
        sims[i] = x
    
    return sims

def garch(omega, alpha1, beta1, nSteps, nSims):
    """
    GARCH(1,1) process simulation.
    
    Parameters:
        omega: constant term
        alpha1: ARCH coefficient
        beta1: GARCH coefficient
        nSteps: number of time steps
        nSims: number of simulations
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        x = np.zeros(nSteps)
        sigma2 = np.zeros(nSteps)
        sigma2[0] = omega / (1 - alpha1 - beta1) if (alpha1 + beta1) < 1 else 1.0
        
        for t in range(nSteps):
            eps = np.random.randn()
            x[t] = eps * np.sqrt(sigma2[t])
            if t < nSteps - 1:
                sigma2[t + 1] = omega + alpha1 * x[t]**2 + beta1 * sigma2[t]
        
        sims[i] = x
    
    return sims

def heston(mu, kappa, theta, sigma, rho, s0=100, v0=0.04, nSteps=252, nSims=1000, dt=1/252):
    """
    Heston stochastic volatility model simulation.
    
    Parameters:
        mu: drift of stock price
        kappa: mean reversion rate of variance
        theta: long-term variance
        sigma: volatility of variance
        rho: correlation between stock and variance
        s0: initial stock price
        v0: initial variance
        nSteps: number of time steps
        nSims: number of simulations
        dt: time step
    
    Returns:
        dict with 'prices' and 'variances' arrays (nSims x nSteps)
    """
    prices = np.zeros((nSims, nSteps))
    variances = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        S = np.zeros(nSteps)
        V = np.zeros(nSteps)
        
        S[0] = s0
        V[0] = v0
        
        for t in range(1, nSteps):
            Z1 = np.random.randn()
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn()
            
            V[t] = max(V[t - 1] + kappa * (theta - V[t - 1]) * dt + 
                      sigma * np.sqrt(V[t - 1] * dt) * Z2, 0)
            
            S[t] = S[t - 1] * np.exp((mu - 0.5 * V[t - 1]) * dt + 
                                      np.sqrt(V[t - 1] * dt) * Z1)
        
        prices[i] = S
        variances[i] = V
    
    return {'prices': prices, 'variances': variances}

def cir(kappa, theta, sigma, nSteps, nSims, r0=0.05, dt=1/252):
    """
    Cox-Ingersoll-Ross (CIR) interest rate model simulation.
    
    Parameters:
        kappa: mean reversion rate
        theta: long-term mean
        sigma: volatility
        nSteps: number of time steps
        nSims: number of simulations
        r0: initial rate
        dt: time step
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        r = np.zeros(nSteps)
        r[0] = r0
        
        for t in range(1, nSteps):
            dW = np.sqrt(dt) * np.random.randn()
            dr = kappa * (theta - r[t - 1]) * dt + sigma * np.sqrt(max(r[t - 1], 0)) * dW
            r[t] = max(r[t - 1] + dr, 0)
        
        sims[i] = r
    
    return sims

def vasicek(kappa, theta, sigma, nSteps, nSims, r0=0.05, dt=1/252):
    """
    Vasicek interest rate model simulation.
    
    Parameters:
        kappa: mean reversion rate
        theta: long-term mean
        sigma: volatility
        nSteps: number of time steps
        nSims: number of simulations
        r0: initial rate
        dt: time step
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        r = np.zeros(nSteps)
        r[0] = r0
        
        for t in range(1, nSteps):
            dW = np.sqrt(dt) * np.random.randn()
            dr = kappa * (theta - r[t - 1]) * dt + sigma * dW
            r[t] = r[t - 1] + dr
        
        sims[i] = r
    
    return sims

def poisson(lambdaRate, nSteps, nSims):
    """
    Poisson process simulation (jump counts).
    
    Parameters:
        lambdaRate: jump intensity (jumps per period)
        nSteps: number of time steps
        nSims: number of simulations
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps), dtype=int)
    
    for i in range(nSims):
        jumps = np.random.poisson(lambdaRate, nSteps)
        sims[i] = np.cumsum(jumps)
    
    return sims

def compoundPoisson(lambdaRate, jumpMu, jumpSigma, nSteps, nSims, s0=100, dt=1/252):
    """
    Compound Poisson process simulation (jumps with sizes).
    
    Parameters:
        lambdaRate: jump intensity
        jumpMu: mean jump size
        jumpSigma: jump size std dev
        nSteps: number of time steps
        nSims: number of simulations
        s0: initial value
        dt: time step
    
    Returns:
        array (nSims x nSteps)
    """
    sims = np.zeros((nSims, nSteps))
    
    for i in range(nSims):
        S = np.zeros(nSteps)
        S[0] = s0
        
        for t in range(1, nSteps):
            nJumps = np.random.poisson(lambdaRate * dt)
            if nJumps > 0:
                jumpSizes = np.random.normal(jumpMu, jumpSigma, nJumps)
                totalJump = np.sum(jumpSizes)
            else:
                totalJump = 0
            
            S[t] = S[t - 1] + totalJump
        
        sims[i] = S
    
    return sims

def simulate(model, params, nSteps, nSims):
    """
    General simulation dispatcher.
    
    Parameters:
        model: model name (str)
        params: dict of model parameters
        nSteps: number of time steps
        nSims: number of simulations
    
    Returns:
        array of simulated paths
    """
    model = model.lower()
    
    if model == 'gbm':
        return gbm(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'ou':
        return ou(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'levyou' or model == 'levy_ou':
        return levyOu(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'ar1':
        return ar1(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'arma':
        return arma(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'markovswitching' or model == 'markov_switching':
        return markovSwitching(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'arch':
        return arch(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'garch':
        return garch(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'heston':
        return heston(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'cir':
        return cir(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'vasicek':
        return vasicek(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'poisson':
        return poisson(nSteps=nSteps, nSims=nSims, **params)
    elif model == 'compoundpoisson' or model == 'compound_poisson':
        return compoundPoisson(nSteps=nSteps, nSims=nSims, **params)
    else:
        raise ValueError(f"Unknown model: {model}")