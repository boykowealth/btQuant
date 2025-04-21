import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy import stats

def fit_gbm(prices, dt=1/252):
    """
    Fits a Geometric Brownian Motion (GBM) model to the provided price data.

    The GBM model is commonly used in finance to model stock prices and other financial assets. 
    It assumes that the price follows a continuous-time stochastic process with constant drift and volatility.

    Parameters:
        prices (array-like): A 1D array of price data.
        dt (float): The time step for the model (default is 1/252, which represents daily data assuming 252 trading days in a year).

    Returns:
        dict: A dictionary containing the estimated parameters of the GBM model:
            - 'mu': The drift term (expected return).
            - 'sigma': The volatility (standard deviation of returns).
    """
    log_returns = np.diff(np.log(prices))
    mu = np.mean(log_returns) / dt + 0.5 * np.var(log_returns) / dt
    sigma = np.std(log_returns) / np.sqrt(dt)
    return {"mu": mu, "sigma": sigma}

def fit_ou(spread, dt=1/252):
    """
    Fits an Ornstein-Uhlenbeck (OU) process to the provided spread data.

    The OU process is commonly used to model mean-reverting processes, often applied to financial spreads or interest rates.
    It describes a process that reverts to a long-term mean with a constant speed of mean reversion and volatility.

    Parameters:
        spread (array-like): A 1D array of spread data (the variable being modeled).
        dt (float): The time step for the model (default is 1/252, which represents daily data assuming 252 trading days in a year).

    Returns:
        dict: A dictionary containing the estimated parameters of the OU model:
            - 'theta': The mean reversion rate.
            - 'mu': The long-term mean level.
            - 'sigma': The volatility of the process.
            - 'half_life': The time it takes for the process to revert halfway to the mean.
    """
    n = len(spread)
    Sx = np.sum(spread[:-1])
    Sy = np.sum(spread[1:])
    Sxx = np.sum(spread[:-1] ** 2)
    Syy = np.sum(spread[1:] ** 2)
    Sxy = np.sum(spread[:-1] * spread[1:])
    
    mu = (Sy * Sxx - Sx * Sxy) / ((n - 1) * (Sxx - Sxy) - (Sx**2 - Sx * Sy))
    theta = -np.log((Sxy - mu * Sx - mu * Sy + (n - 1) * mu ** 2) / (Sxx - 2 * mu * Sx + (n - 1) * mu ** 2))
    a = np.exp(-theta)
    sigmah2 = (Syy - 2 * a * Sxy + a ** 2 * Sxx - 2 * mu * (1 - a) * (Sy - a * Sx) + (n - 1) * mu ** 2 * (1 - a) ** 2) / (n - 1)
    sigma = np.sqrt(sigmah2 * 2 * theta / (1 - a ** 2))
    half_life = np.log(2) / theta / dt
    return {"theta": theta / dt, "mu": mu, "sigma": sigma * np.sqrt(1 / dt), "half_life": half_life}


def fit_levy_ou(spread, dt=1/252):
    """
    Fits a Lévy Ornstein-Uhlenbeck (OU) model, also known as Jump Diffusion OU, to the provided spread data.
    This model extends the standard OU process by adding jumps to account for sudden, large movements in the spread.

    The model assumes that the process can be described by a combination of a mean-reverting OU process and jumps, where 
    the jumps follow a normal distribution with a specified mean and standard deviation.

    Parameters:
        spread (array-like): A 1D array of spread data (the variable being modeled).
        dt (float): The time step for the model (default is 1/252, representing daily data assuming 252 trading days in a year).

    Returns:
        dict: A dictionary containing the estimated parameters of the Lévy OU model:
            - 'theta': The mean reversion rate.
            - 'mu': The long-term mean level.
            - 'sigma': The volatility of the process.
            - 'half_life': The time it takes for the process to revert halfway to the mean.
            - 'jump_lambda': The intensity of jumps (probability of a jump per time step).
            - 'jump_mu': The mean of the jump size.
            - 'jump_sigma': The standard deviation of the jump size.
    """

    def neg_log_likelihood(params):
        theta, mu, sigma, lam, jump_mu, jump_sigma = params
        X = spread[:-1]
        Y = spread[1:]
        drift = X + (mu - X) * (1 - np.exp(-theta * dt))
        variance = sigma**2 * (1 - np.exp(-2 * theta * dt)) / (2 * theta)
        jump_term = lam * dt * norm.pdf(Y - drift, loc=jump_mu, scale=np.sqrt(variance + jump_sigma ** 2))
        no_jump_term = (1 - lam * dt) * norm.pdf(Y, loc=drift, scale=np.sqrt(variance))
        likelihood = no_jump_term + jump_term
        return -np.sum(np.log(likelihood + 1e-10))

    init_params = [1.0, np.mean(spread), np.std(spread), 0.1, 0.0, 0.1]
    bounds = [(1e-4, 10), (-np.inf, np.inf), (1e-4, np.inf), (1e-4, 1), (-np.inf, np.inf), (1e-4, np.inf)]
    result = minimize(neg_log_likelihood, init_params, bounds=bounds)
    theta, mu, sigma, lam, jump_mu, jump_sigma = result.x
    half_life = np.log(2)/theta/dt
    return {
        "theta": theta / dt,
        "mu": mu,
        "sigma": sigma * np.sqrt(1 / dt),
        "half_life": half_life,
        "jump_lambda": lam,
        "jump_mu": jump_mu,
        "jump_sigma": jump_sigma
    }

def fit_ar1(series):
    """
    Fits an AR(1) (AutoRegressive of order 1) model to the given time series data.
    The AR(1) model assumes the value of the series at time t is a linear function of the value at time t-1 
    plus a noise term.

    The model is of the form: 
        series[t] = intercept + ar1_coefficient * series[t-1] + noise[t]

    Parameters:
        series (array-like): A 1D array or list containing the time series data.

    Returns:
        dict: A dictionary containing the estimated parameters of the AR(1) model:
            - 'ar1_coefficient': The coefficient of the lag-1 term.
            - 'intercept': The intercept of the model (constant term).
            - 'sigma2': The estimated variance of the residuals (error term).
    """
    
    series = np.array(series)
    intercept = np.mean(series)
    series_demeaned = series - intercept

    ar1_coefficient = np.sum(series_demeaned[:-1] * series_demeaned[1:]) / np.sum(series_demeaned[:-1]**2)
    residuals = series_demeaned[1:] - ar1_coefficient * series_demeaned[:-1]
    sigma2 = np.var(residuals, ddof=1)
    
    return {
        "ar1_coefficient": ar1_coefficient,
        "intercept": intercept,
        "sigma2": sigma2
    }

def fit_arima(series, order=(1, 1, 1)):
    """
    Fits an ARIMA model to the given time series data.
    The ARIMA model has three components:
        - AR(p): AutoRegressive part of order p
        - I(d): Integrated part (differencing) of order d
        - MA(q): Moving Average part of order q

    Parameters:
        series (array-like): A 1D array or list containing the time series data.
        order (tuple): A tuple of the form (p, d, q) where:
            - p: the order of the AR term
            - d: the order of differencing
            - q: the order of the MA term

    Returns:
        dict: A dictionary containing:
            - 'params': The model parameters (AR, MA coefficients, etc.)
            - 'aic': Akaike Information Criterion for model fit
            - 'bic': Bayesian Information Criterion for model fit
    """
    
    series = np.array(series)
    diff_series = series.copy()
    for _ in range(order[1]):
        diff_series = np.diff(diff_series)
    
    p = order[0]
    ar_params = np.zeros(p)
    for i in range(p, len(diff_series)):
        ar_params[i-p:i] = np.linalg.lstsq(
            np.vstack([diff_series[i-p:i]]).T, diff_series[i], rcond=None)[0]
    
    q = order[2]
    ma_params = np.zeros(q)
    for i in range(q, len(diff_series)):
        ma_params[i-q:i] = np.linalg.lstsq(
            np.vstack([diff_series[i-q:i]]).T, diff_series[i], rcond=None)[0]

    residuals = diff_series - np.dot(np.vstack([ar_params, ma_params]), diff_series.T)

    n = len(residuals)
    rss = np.sum(residuals**2)
    aic = n * np.log(rss/n) + 2 * (p + q + 1)
    bic = n * np.log(rss/n) + np.log(n) * (p + q + 1)

    return {
        "params": {
            "ar_params": ar_params.tolist(),
            "ma_params": ma_params.tolist()
        },
        "aic": aic,
        "bic": bic
    }

def fit_markov_switching(series, k_regimes=2, max_iter=100, tol=1e-6):
    """
    Fits a Markov Switching model to the given time series data.
    This model assumes 'k_regimes' number of regimes (hidden states).
    
    Parameters:
        series (array-like): A 1D array or list containing the time series data.
        k_regimes (int): Number of hidden regimes (states) in the Markov model.
        max_iter (int): Maximum number of iterations for the EM algorithm.
        tol (float): Convergence tolerance for the EM algorithm.
        
    Returns:
        dict: A dictionary containing:
            - 'params': The model parameters (mean, variance, and transition probabilities for each regime).
            - 'smoothed_probs': The smoothed marginal probabilities for each regime at each time step.
            - 'llf': The log-likelihood of the fitted model.
    """

    series = np.array(series)
    n = len(series)
    
    means = np.random.randn(k_regimes)
    variances = np.random.rand(k_regimes)
    transition_probs = np.ones((k_regimes, k_regimes)) / k_regimes

    state_probs = np.random.rand(n, k_regimes)
    state_probs /= state_probs.sum(axis=1, keepdims=True)
    
    log_likelihood = 0
    prev_log_likelihood = -np.inf
    
    for iteration in range(max_iter):
        
        forward_probs = np.zeros((n, k_regimes))
        backward_probs = np.zeros((n, k_regimes)) 
        
        for state in range(k_regimes):
            forward_probs[0, state] = (1 / np.sqrt(2 * np.pi * variances[state])) * \
                                      np.exp(-(series[0] - means[state])**2 / (2 * variances[state]))
        
        forward_probs[0] /= forward_probs[0].sum()
        
        for t in range(1, n):
            for state in range(k_regimes):
                forward_probs[t, state] = np.sum(forward_probs[t-1] * transition_probs[:, state]) * \
                                          (1 / np.sqrt(2 * np.pi * variances[state])) * \
                                          np.exp(-(series[t] - means[state])**2 / (2 * variances[state]))
            forward_probs[t] /= forward_probs[t].sum()
        
        backward_probs[-1] = 1 
        
        for t in range(n-2, -1, -1):
            for state in range(k_regimes):
                backward_probs[t, state] = np.sum(transition_probs[state] * backward_probs[t+1] *
                                                   (1 / np.sqrt(2 * np.pi * variances) *
                                                    np.exp(-(series[t+1] - means) ** 2 / (2 * variances))))
            backward_probs[t] /= backward_probs[t].sum()
        
        smoothed_probs = forward_probs * backward_probs
        smoothed_probs /= smoothed_probs.sum(axis=1, keepdims=True) 
        
        transition_probs = np.zeros((k_regimes, k_regimes))
        for t in range(n-1):
            for current_state in range(k_regimes):
                for next_state in range(k_regimes):
                    transition_probs[current_state, next_state] += smoothed_probs[t, current_state] * \
                                                                    forward_probs[t+1, next_state] / \
                                                                    np.sum(forward_probs[t])
        
        transition_probs /= transition_probs.sum(axis=1, keepdims=True)
        
        for state in range(k_regimes):
            weight = smoothed_probs[:, state]
            means[state] = np.sum(weight * series) / np.sum(weight)
            variances[state] = np.sum(weight * (series - means[state])**2) / np.sum(weight)

        ll = np.sum(np.log(np.sum(forward_probs, axis=1)))

        if np.abs(ll - prev_log_likelihood) < tol:
            break
        
        prev_log_likelihood = ll
    
    return {
        "params": {
            "means": means.tolist(),
            "variances": variances.tolist(),
            "transition_probs": transition_probs.tolist()
        },
        "smoothed_probs": smoothed_probs.tolist(),
        "llf": ll
    }

def fit_arch(series, p=1):
    """
    Fits an ARCH(p) model to the given time series data.
    
    Parameters:
        series (array-like): A 1D array or list containing the time series data.
        p (int): The order of the ARCH model (number of lags).
        
    Returns:
        dict: A dictionary containing:
            - 'params': The model parameters (intercept and coefficients for the lagged squared returns).
            - 'aic': The Akaike Information Criterion for the fitted model.
            - 'bic': The Bayesian Information Criterion for the fitted model.
    """
    
    series = np.array(series)
    n = len(series)
    
    init_params = np.ones(p + 1) * 0.1 
    
    def log_likelihood(params):
        alpha0 = params[0] 
        alphas = params[1:]
        sigma2 = np.zeros(n)
        
        sigma2[p:] = alpha0 + np.sum(alphas * (series[p-1:n-1])**2, axis=1)
        sigma2 = np.maximum(sigma2, 1e-6)
        
        log_likelihood_val = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (series[p:]**2) / sigma2)
        
        return -log_likelihood_val 
    
    result = minimize(log_likelihood, init_params, bounds=[(1e-6, None)] * (p + 1))
    
    params = result.x
    alpha0 = params[0] 
    alphas = params[1:] 
    
    sigma2 = np.zeros(n)
    sigma2[p:] = alpha0 + np.sum(alphas * (series[p-1:n-1])**2, axis=1)
    sigma2 = np.maximum(sigma2, 1e-6)
    
    log_likelihood_val = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + (series[p:]**2) / sigma2)
    
    aic = 2 * (p + 1) - 2 * log_likelihood_val
    bic = np.log(n) * (p + 1) - 2 * log_likelihood_val

    return {
        "params": {"alpha0": alpha0, **{f"alpha{i+1}": alphas[i] for i in range(p)}},
        "aic": aic,
        "bic": bic
    }

def fit_garch(series, p=1, q=1):
    """
    Fits a GARCH(p, q) model to the given time series data.
    
    Parameters:
        series (array-like): A 1D array or list containing the time series data.
        p (int): The order of the ARCH part (number of lagged squared errors).
        q (int): The order of the GARCH part (number of lagged variances).
        
    Returns:
        dict: A dictionary containing:
            - 'params': The model parameters (alpha0, alpha1, ..., alpha_p for ARCH, beta1, ..., beta_q for GARCH).
            - 'aic': The Akaike Information Criterion for the fitted model.
            - 'bic': The Bayesian Information Criterion for the fitted model.
    """
    
    series = np.array(series)
    n = len(series)
    
    init_params = np.ones(p + q + 1) * 0.1 
    
    def log_likelihood(params):
        alpha0 = params[0] 
        alphas = params[1:p+1]
        betas = params[p+1:]
        
        sigma2 = np.zeros(n)
        epsilon2 = series**2 
        
        sigma2[p + q:] = alpha0 + np.sum(alphas * epsilon2[p-q-1:n-q-1], axis=1) + np.sum(betas * sigma2[p-q-1:n-q-1], axis=1)
        sigma2 = np.maximum(sigma2, 1e-6)
        
        log_likelihood_val = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + epsilon2[p+q:] / sigma2)
        
        return -log_likelihood_val
    
    result = minimize(log_likelihood, init_params, bounds=[(1e-6, None)] * (p + q + 1))

    params = result.x
    alpha0 = params[0]
    alphas = params[1:p+1]
    betas = params[p+1:]
    
    sigma2 = np.zeros(n)
    epsilon2 = series**2
    
    sigma2[p + q:] = alpha0 + np.sum(alphas * epsilon2[p-q-1:n-q-1], axis=1) + np.sum(betas * sigma2[p-q-1:n-q-1], axis=1)
    sigma2 = np.maximum(sigma2, 1e-6)
    
    log_likelihood_val = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + epsilon2[p+q:] / sigma2)
    aic = 2 * (p + q + 1) - 2 * log_likelihood_val
    bic = np.log(n) * (p + q + 1) - 2 * log_likelihood_val
    
    return {
        "params": {"alpha0": alpha0, **{f"alpha{i+1}": alphas[i] for i in range(p)}, **{f"beta{i+1}": betas[i] for i in range(q)}},
        "aic": aic,
        "bic": bic
    }

def fit_distributions(data, distributions=['norm', 't', 'laplace', 'lognorm', 'beta', 'skewnorm', 'gamma', 'genpareto', 'genextreme', 'cauchy', 'invgauss', 'loggamma', 'exponweib']):
    results = {}
    for dist_name in distributions:
        dist = getattr(stats, dist_name)
        params = dist.fit(data)
        log_likelihood = np.sum(dist.logpdf(data, *params))
        aic = 2 * len(params) - 2 * log_likelihood
        results[dist_name] = {"params": params, "aic": aic}
    return sorted(results.items(), key=lambda x: x[1]["aic"])

