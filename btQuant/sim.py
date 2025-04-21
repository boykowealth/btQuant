import numpy as np
import pandas as pd
from numpy.random import normal, poisson

def sim_gbm(mu, sigma, n_steps, n_sim, s0=1.0, dt=1/252):
    sims = []
    for _ in range(n_sim):
        shocks = normal(loc=(mu - 0.5 * sigma ** 2) * dt, scale=sigma * np.sqrt(dt), size=n_steps)
        path = s0 * np.exp(np.cumsum(shocks))
        sims.append(path)
    return np.array(sims)

def sim_ou(theta, mu, sigma, n_steps, n_sim, x0=0.0, dt=1/252):
    sims = []
    for _ in range(n_sim):
        x = [x0]
        for _ in range(n_steps - 1):
            dx = theta * (mu - x[-1]) * dt + sigma * np.sqrt(dt) * np.random.randn()
            x.append(x[-1] + dx)
        sims.append(x)
    return np.array(sims)

def sim_levy_ou(theta, mu, sigma, jump_lambda, jump_mu, jump_sigma, n_steps, n_sim, x0=0.0, dt=1/252):
    sims = []
    for _ in range(n_sim):
        x = [x0]
        for _ in range(n_steps - 1):
            jump = 0
            if np.random.rand() < jump_lambda * dt:
                jump = np.random.normal(jump_mu, jump_sigma)
            dx = theta * (mu - x[-1]) * dt + sigma * np.sqrt(dt) * np.random.randn() + jump
            x.append(x[-1] + dx)
        sims.append(x)
    return np.array(sims)

def sim_ar1(phi, intercept, sigma, n_steps, n_sim, x0=0.0):
    sims = []
    for _ in range(n_sim):
        x = [x0]
        for _ in range(n_steps - 1):
            x.append(intercept + phi * x[-1] + np.random.normal(0, np.sqrt(sigma)))
        sims.append(x)
    return np.array(sims)

def sim_arima(ar, ma, d, sigma, n_steps, n_sim, x0=0.0):
    from statsmodels.tsa.arima_process import ArmaProcess
    sims = []
    ar_params = np.r_[1, -ar]
    ma_params = np.r_[1, ma]
    arma = ArmaProcess(ar_params, ma_params)
    for _ in range(n_sim):
        series = arma.generate_sample(nsample=n_steps, scale=np.sqrt(sigma))
        if d > 0:
            series = np.cumsum(series)
        sims.append(series + x0)
    return np.array(sims)

def sim_markov_switching(mu1, sigma1, mu2, sigma2, p11, p22, n_steps, n_sim, x0=0.0):
    sims = []
    for _ in range(n_sim):
        x = [x0]
        state = 0  # start in regime 0
        for _ in range(n_steps - 1):
            if state == 0:
                x.append(x[-1] + np.random.normal(mu1, sigma1))
                state = 0 if np.random.rand() < p11 else 1
            else:
                x.append(x[-1] + np.random.normal(mu2, sigma2))
                state = 1 if np.random.rand() < p22 else 0
        sims.append(x)
    return np.array(sims)

def sim_arch(alpha0, alpha1, n_steps, n_sim):
    sims = []
    for _ in range(n_sim):
        x = [0]
        sigma2 = [alpha0 / (1 - alpha1)]
        for _ in range(n_steps - 1):
            eps = np.random.randn()
            sigma2_t = alpha0 + alpha1 * x[-1] ** 2
            x_t = eps * np.sqrt(sigma2_t)
            x.append(x_t)
            sigma2.append(sigma2_t)
        sims.append(x)
    return np.array(sims)

def sim_garch(omega, alpha1, beta1, n_steps, n_sim):
    sims = []
    for _ in range(n_sim):
        x = [0]
        sigma2 = [omega / (1 - alpha1 - beta1)]
        for _ in range(n_steps - 1):
            eps = np.random.randn()
            sigma2_t = omega + alpha1 * x[-1] ** 2 + beta1 * sigma2[-1]
            x_t = eps * np.sqrt(sigma2_t)
            x.append(x_t)
            sigma2.append(sigma2_t)
        sims.append(x)
    return np.array(sims)

def sim(model: str, params: dict, n_steps: int, n_sim: int) -> pd.DataFrame:
    """
    Simulates various time series models and returns the simulated paths as a pandas DataFrame.

    Parameters:
        model (str): The name of the model to simulate. Possible values are 'gbm', 'ou', 'levy_ou', 'ar1', 
                    'arima', 'markov_switching', 'arch', 'garch'.
        params (dict): A dictionary containing the parameters for the selected model.
        n_steps (int): The number of time steps in each simulation.
        n_sim (int): The number of simulations to generate.

    Returns:
        pd.DataFrame: A DataFrame where each column represents one simulation, and each row represents one time step.
    """

    model = model.lower()
    
    if model == 'gbm':
        data = sim_gbm(**params, n_steps=n_steps, n_sim=n_sim)
    elif model == 'ou':
        data = sim_ou(**params, n_steps=n_steps, n_sim=n_sim)
    elif model == 'levy_ou':
        data = sim_levy_ou(**params, n_steps=n_steps, n_sim=n_sim)
    elif model == 'ar1':
        data = sim_ar1(**params, n_steps=n_steps, n_sim=n_sim)
    elif model == 'arima':
        data = sim_arima(**params, n_steps=n_steps, n_sim=n_sim)
    elif model == 'markov_switching':
        data = sim_markov_switching(**params, n_steps=n_steps, n_sim=n_sim)
    elif model == 'arch':
        data = sim_arch(**params, n_steps=n_steps, n_sim=n_sim)
    elif model == 'garch':
        data = sim_garch(**params, n_steps=n_steps, n_sim=n_sim)
    else:
        raise ValueError(f"Unknown model: {model}")
    
    df = pd.DataFrame(data.T, columns=[f"sim_{i+1}" for i in range(n_sim)])
    return df