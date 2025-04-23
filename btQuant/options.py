import numpy as np
from scipy.stats import norm
from itertools import product
import pandas as pd

def blackScholes(S, K, T, r, sigma, b=None, option_type='call'):
    """
    Prices a European option using the Black-Scholes model with cost of carry.
    Parameters:
        S : float - current stock price
        K : float - strike price
        T : float - time to maturity (in years)
        r : float - risk-free rate
        sigma : float - volatility
        b : float - cost of carry (if None, defaults to r)
            - For stocks with dividend yield q: b = r - q
            - For futures: b = 0
            - For indices with continuous dividend yield q: b = r - q
            - For currencies with foreign risk-free rate rf: b = r - rf
        option_type : str - 'call' or 'put'
    Returns:
        dict with price, delta, gamma, vega, rho, theta
    """
    if b is None:
        b = r
    
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * np.exp((b - r) * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp((b - r) * T) * norm.cdf(d1)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp((b - r) * T) * norm.cdf(-d1)
        delta = -np.exp((b - r) * T) * norm.cdf(-d1)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    gamma = np.exp((b - r) * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.exp((b - r) * T) * norm.pdf(d1) * np.sqrt(T)
    
    if option_type == 'call':
        theta = -S * np.exp((b - r) * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - (b - r) * S * np.exp((b - r) * T) * norm.cdf(d1) - r * K * np.exp(-r * T) * norm.cdf(d2)
    else:
        theta = -S * np.exp((b - r) * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + (b - r) * S * np.exp((b - r) * T) * norm.cdf(-d1) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta
    }

def binomialTree(S, K, T, r, sigma, b=None, N=100, option_type='call', american=False):
    """
    Prices European or American options using the Binomial Tree method with cost of carry.
    Parameters:
        S : float - current stock price
        K : float - strike price
        T : float - time to maturity (in years)
        r : float - risk-free rate
        sigma : float - volatility
        b : float - cost of carry (if None, defaults to r)
            - For stocks with dividend yield q: b = r - q
            - For futures: b = 0
            - For indices with continuous dividend yield q: b = r - q
            - For currencies with foreign risk-free rate rf: b = r - rf
        N : int - number of time steps
        option_type : str - 'call' or 'put'
        american : bool - whether the option is American style
    Returns:
        dict with price, delta, gamma, theta
    """
    if b is None:
        b = r
    
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    
    q = (np.exp(b * dt) - d) / (u - d)
    
    S_T = np.zeros((N+1, N+1))
    option_values = np.zeros((N+1, N+1))
    
    for i in range(N+1):
        for j in range(i+1):
            S_T[j, i] = S * (u**(i-j)) * (d**j)
            
    for j in range(N+1):
        if option_type == 'call':
            option_values[j, N] = max(0, S_T[j, N] - K)
        else:
            option_values[j, N] = max(0, K - S_T[j, N])
    
    for i in range(N-1, -1, -1):
        for j in range(i+1):
            option_values[j, i] = np.exp(-r * dt) * (q * option_values[j, i+1] + (1-q) * option_values[j+1, i+1])
            if american:
                if option_type == 'call':
                    option_values[j, i] = max(option_values[j, i], S_T[j, i] - K)
                else:
                    option_values[j, i] = max(option_values[j, i], K - S_T[j, i])
    
    price = option_values[0, 0]
    
    if N > 1:
        delta = (option_values[0, 1] - option_values[1, 1]) / (S_T[0, 1] - S_T[1, 1])
        if N > 2:
            gamma = ((option_values[0, 2] - option_values[1, 2]) / (S_T[0, 2] - S_T[1, 2]) - 
                    (option_values[1, 2] - option_values[2, 2]) / (S_T[1, 2] - S_T[2, 2])) / ((S_T[0, 2] - S_T[2, 2])/2)
        else:
            gamma = 0
    else:
        delta = 0
        gamma = 0
    
    theta = (option_values[0, 1] - price) / dt
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta
    }

def trinomialTree(S, K, T, r, sigma, b=None, N=50, option_type='call', american=False):
    """
    Prices European or American options using the Trinomial Tree method with cost of carry.
    Parameters:
        S : float - current stock price
        K : float - strike price
        T : float - time to maturity (in years)
        r : float - risk-free rate
        sigma : float - volatility
        b : float - cost of carry (if None, defaults to r)
            - For stocks with dividend yield q: b = r - q
            - For futures: b = 0
            - For indices with continuous dividend yield q: b = r - q
            - For currencies with foreign risk-free rate rf: b = r - rf
        N : int - number of time steps
        option_type : str - 'call' or 'put'
        american : bool - whether the option is American style
        calculate_greeks : bool - whether to calculate Greeks
    Returns:
        dict with price, delta, gamma, theta
    """
    if b is None:
        b = r
    
    def compute_price(S_val, T_val):
        dt = T_val / N
        dx = sigma * np.sqrt(3 * dt)
        
        pu = 1/6 + (b - 0.5 * sigma**2) * dt / (2 * dx)
        pd = 1/6 - (b - 0.5 * sigma**2) * dt / (2 * dx)
        pm = 2/3
        
        df = np.exp(-r * dt)
        
        asset_prices = np.zeros(2*N+1)
        for i in range(2*N+1):
            asset_prices[i] = S_val * np.exp((i-N) * dx)
        
        if option_type == 'call':
            option_values = np.maximum(asset_prices - K, 0)
        else:
            option_values = np.maximum(K - asset_prices, 0)
        
        for i in range(N-1, -1, -1):
            new_option_values = np.zeros(2*i+1)
            for j in range(2*i+1):
                idx = j - i + N 
                new_option_values[j] = df * (pu * option_values[idx+2] + pm * option_values[idx+1] + pd * option_values[idx])
                
                if american:
                    current_price = S_val * np.exp((j-i) * dx)
                    if option_type == 'call':
                        new_option_values[j] = max(new_option_values[j], current_price - K)
                    else:
                        new_option_values[j] = max(new_option_values[j], K - current_price)
            
            option_values = new_option_values
        
        return option_values[0]
    
    price = compute_price(S, T)
    
    delta = None
    gamma = None
    theta = None
    
    h = 0.01 * S
    
    price_up = compute_price(S+h, T)
    price_down = compute_price(S-h, T)
    
    delta = (price_up - price_down) / (2 * h)
    gamma = (price_up - 2*price + price_down) / (h**2)
    
    dt = T / N
    if T > dt:
        price_dt = compute_price(S, T-dt)
        theta = (price_dt - price) / dt
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'theta': theta
    }

def asian(S, K, T, r, sigma, b=None, n_steps=100, option_type='call'):
    """
    Prices Asian options with geometric averaging using analytic formula with cost of carry.
    Parameters:
        S : float - current stock price
        K : float - strike price
        T : float - time to maturity (in years)
        r : float - risk-free rate
        sigma : float - volatility
        b : float - cost of carry (if None, defaults to r)
            - For stocks with dividend yield q: b = r - q
            - For futures: b = 0
            - For indices with continuous dividend yield q: b = r - q
            - For currencies with foreign risk-free rate rf: b = r - rf
        n_steps : int - number of time steps in simulation
        n_simulations : int - number of simulations
        option_type : str - 'call' or 'put'
    Returns:
        dict with price, delta, gamma, vega, rho, theta
    """
    if b is None:
        b = r
    
    def compute_price(S_val, T_val, r_val=r, sigma_val=sigma, b_val=b):
        dt = T_val / n_steps
        
        sigma_adj = sigma_val * np.sqrt((n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps**2))
        b_adj = (b_val - 0.5 * sigma_val**2) * (n_steps + 1) / (2 * n_steps) + 0.5 * sigma_adj**2
        
        d1 = (np.log(S_val / K) + (b_adj + 0.5 * sigma_adj**2) * T_val) / (sigma_adj * np.sqrt(T_val))
        d2 = d1 - sigma_adj * np.sqrt(T_val)
        
        if option_type == 'call':
            price = np.exp(-r_val * T_val) * (S_val * np.exp(b_adj * T_val) * norm.cdf(d1) - K * norm.cdf(d2))
        else:
            price = np.exp(-r_val * T_val) * (K * norm.cdf(-d2) - S_val * np.exp(b_adj * T_val) * norm.cdf(-d1))
            
        return price
    
    price = compute_price(S, T)
    
    dt = T / n_steps
    sigma_adj = sigma * np.sqrt((n_steps + 1) * (2 * n_steps + 1) / (6 * n_steps**2))
    b_adj = (b - 0.5 * sigma**2) * (n_steps + 1) / (2 * n_steps) + 0.5 * sigma_adj**2
    
    d1 = (np.log(S / K) + (b_adj + 0.5 * sigma_adj**2) * T) / (sigma_adj * np.sqrt(T))
    
    if option_type == 'call':
        delta = np.exp(-r * T + b_adj * T) * norm.cdf(d1)
    else:
        delta = -np.exp(-r * T + b_adj * T) * norm.cdf(-d1)
    
    h = 0.01 * S
    
    price_up = compute_price(S+h, T)
    price_down = compute_price(S-h, T)
    
    
    gamma = (price_up - 2*price + price_down) / (h**2)
    
    h_vol = 0.005
    price_vol_up = compute_price(S, T, r, sigma+h_vol, b)
    price_vol_down = compute_price(S, T, r, sigma-h_vol, b)
    vega = (price_vol_up - price_vol_down) / (2 * h_vol)
    
    if T > dt:
        price_dt = compute_price(S, T-dt)
        theta = (price_dt - price) / dt
    else:
        theta = None
    
    h_r = 0.0025
    price_r_up = compute_price(S, T, r+h_r, sigma, b)
    price_r_down = compute_price(S, T, r-h_r, sigma, b)
    rho = (price_r_up - price_r_down) / (2 * h_r)
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta
    }

def binary(S, K, T, r, sigma, b=None, option_type='call'):
    """
    Prices a Binary (Cash-or-Nothing) option using the Black-Scholes formula with cost of carry.
    Parameters:
        S : float - current stock price
        K : float - strike price
        T : float - time to maturity (in years)
        r : float - risk-free rate
        sigma : float - volatility
        b : float - cost of carry (if None, defaults to r)
            - For stocks with dividend yield q: b = r - q
            - For futures: b = 0
            - For indices with continuous dividend yield q: b = r - q
            - For currencies with foreign risk-free rate rf: b = r - rf
        option_type : str - 'call' or 'put'
    Returns:
        dict with price, delta, gamma, vega, rho, theta
    """
    if b is None:
        b = r
        
    d1 = (np.log(S / K) + (b + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = np.exp(-r * T) * norm.cdf(d2)
        delta = np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
        rho = -T * price
    else:
        price = np.exp(-r * T) * norm.cdf(-d2)
        delta = -np.exp(-r * T) * norm.pdf(d2) / (S * sigma * np.sqrt(T))
        rho = -T * price
        
    gamma = -np.exp(-r * T) * norm.pdf(d2) * d1 / (S**2 * sigma**2 * T)
    vega = -np.exp(-r * T) * norm.pdf(d2) * d1 / (sigma)
    theta = -np.exp(-r * T) * norm.pdf(d2) * (r + ((-d1 * sigma) / (2 * T * np.sqrt(T))))
    
    return {
        'price': price,
        'delta': delta,
        'gamma': gamma,
        'vega': vega,
        'rho': rho,
        'theta': theta
    }

def real(S, K, T, r, sigma, b=None, dividend=0, option_type='defer'):
    """
    Prices a simple real option (deferral/expansion) using Black-Scholes as a base model with cost of carry.
    Parameters:
        S : float - present value of expected cash flows
        K : float - investment cost/strike price
        T : float - time to expiry of deferral option (in years)
        r : float - risk-free rate
        sigma : float - volatility of project value
        b : float - cost of carry (if None, defaults to r-dividend)
        dividend : float - dividend/leakage rate (opportunity cost of waiting)
        option_type : str - 'defer' (right to wait) or 'expand' (right to scale)
    Returns:
        dict with price (option value), delta, gamma, vega, rho, theta
    """
    if b is None:
        b = r - dividend
    
    if option_type == 'defer':
        return blackScholes(S, K, T, r, sigma, b, 'call')
    elif option_type == 'expand':
        expansion_factor = 1.5
        expansion_value = blackScholes(S * expansion_factor - S, K, T, r, sigma, b, 'call')
        
        base_price = expansion_value['price']
        return {
            'price': base_price,
            'delta': expansion_value['delta'] * expansion_factor,
            'gamma': expansion_value['gamma'] * (expansion_factor**2),
            'vega': expansion_value['vega'] * expansion_factor,
            'rho': expansion_value['rho'],
            'theta': expansion_value['theta']
        }
    else:
        raise ValueError("Option type must be 'defer' or 'expand'")

def generateRange(model_func, param_ranges, fixed_params, option_type='call'):
    """
    Generate a DataFrame of option prices across a range of parameter values.
    
    Parameters:
        model_func : function - the option pricing model function to use
            (e.g. blackScholes, binomialTree, trinomialTree, asian, binary, real)
        
        param_ranges : dict - parameters to vary and their ranges
            Format: {'param_name': {'start': value, 'end': value, 'step': value}}
            Example: {'S': {'start': 90, 'end': 110, 'step': 5}}
        
        fixed_params : dict - parameters to keep fixed
            Format: {'param_name': value}
            Example: {'K': 100, 'T': 1, 'r': 0.05, 'sigma': 0.2}
        
        option_type : str - the type of option ('call', 'put', etc.)
    
    Returns:
        pandas DataFrame with option prices and Greeks
    """
    if not param_ranges:
        raise ValueError("At least one parameter range must be specified")

    param_values = {}
    for param, range_dict in param_ranges.items():
        start = range_dict['start']
        end = range_dict['end']
        step = range_dict['step']
        param_values[param] = np.arange(start, end + step/2, step)
    
    param_names = list(param_values.keys())
    combinations = list(product(*[param_values[p] for p in param_names]))
    
    results = []
    
    for combo in combinations:
        params = fixed_params.copy()
        for i, param_name in enumerate(param_names):
            params[param_name] = combo[i]
        
        params['option_type'] = option_type
        
        try:
            option_result = model_func(**params)
            
            result_row = {param: params[param] for param in param_names}
            
            for key, value in option_result.items():
                result_row[key] = value
                
            results.append(result_row)
        except Exception as e:
            print(f"Error calculating for parameters {params}: {e}")
    
    df = pd.DataFrame(results)

    df = df.sort_values(by=param_names)
    
    return df

# Example usage:
# if __name__ == "__main__":
#     param_ranges_1 = {
#         'S': {'start': 90, 'end': 110, 'step': 5}
#     }
#     fixed_params_1 = {
#         'K': 100, 
#         'T': 1, 
#         'r': 0.05, 
#         'sigma': 0.2
#     }
    
#     df1 = generate_option_price_range(
#         blackScholes, 
#         param_ranges_1, 
#         fixed_params_1, 
#         option_type='call'
#     )
#     print("Black-Scholes prices varying stock price:")
#     print(df1)
    
#     param_ranges_2 = {
#         'S': {'start': 95, 'end': 105, 'step': 5},
#         'T': {'start': 0.5, 'end': 1.5, 'step': 0.5}
#     }
#     fixed_params_2 = {
#         'K': 100, 
#         'r': 0.05, 
#         'sigma': 0.2,
#         'N': 50,
#         'american': True
#     }
    
#     df2 = generate_option_price_range(
#         binomialTree, 
#         param_ranges_2, 
#         fixed_params_2, 
#         option_type='put'
#     )
#     print("\nBinomial Tree prices varying stock price and time to maturity:")
#     print(df2)  