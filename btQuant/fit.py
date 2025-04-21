import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from arch import arch_model

# ===== Geometric Brownian Motion (GBM) Fit ===== #
def fit_gbm(prices, dt=1/252):
    log_returns = np.diff(np.log(prices))
    mu = np.mean(log_returns) / dt + 0.5 * np.var(log_returns) / dt
    sigma = np.std(log_returns) / np.sqrt(dt)
    return {"mu": mu, "sigma": sigma}

# ===== Ornstein-Uhlenbeck (OU) Fit ===== #
def fit_ou(spread, dt=1/252):
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

# ===== Lévy OU Fit (Jump Diffusion OU) ===== #
def fit_levy_ou(spread, dt=1/252):
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

# ===== AR(1) Fit ===== #
def fit_ar1(series):
    model = sm.tsa.ARIMA(series, order=(1, 0, 0))
    result = model.fit()
    return {
        "ar1_coefficient": result.params.get("ar.L1", np.nan),
        "intercept": result.params.get("const", np.nan),
        "sigma2": result.sigma2
    }

# ===== ARIMA Fit ===== #
def fit_arima(series, order=(1, 1, 1)):
    model = ARIMA(series, order=order)
    result = model.fit()
    return {
        "params": result.params.to_dict(),
        "aic": result.aic,
        "bic": result.bic
    }

# ===== Markov Switching Fit ===== #
def fit_markov_switching(series, k_regimes=2):
    model = MarkovRegression(series, k_regimes=k_regimes, trend='c', switching_variance=True)
    result = model.fit()
    return {
        "params": result.params.to_dict(),
        "smoothed_probs": result.smoothed_marginal_probabilities[0].to_list(),
        "llf": result.llf
    }

# ===== ARCH Fit ===== #
def fit_arch(series, p=1):
    model = arch_model(series, vol='ARCH', p=p)
    result = model.fit(disp="off")
    return {
        "params": result.params.to_dict(),
        "aic": result.aic,
        "bic": result.bic
    }

# ===== GARCH Fit ===== #
def fit_garch(series, p=1, q=1):
    model = arch_model(series, vol='GARCH', p=p, q=q)
    result = model.fit(disp="off")
    return {
        "params": result.params.to_dict(),
        "aic": result.aic,
        "bic": result.bic
    }
