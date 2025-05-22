import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import norm, multivariate_normal
from scipy.optimize import root_scalar
from typing import Tuple, Dict
import warnings
import plotly.graph_objects as go

def identify(data: np.ndarray, 
             alpha: float = 0.05,
             return_all: bool = False) -> Tuple[str, Dict]:
    """
    Identify the best fitting distribution for the given data.
    
    Parameters:
    -----------
    data : np.ndarray
        1D array of data to fit
    alpha : float, default=0.05
        Significance level for statistical tests
    return_all : bool, default=False
        If True, return results for all distributions tested
        
    Returns:
    --------
    best_dist : str
        Name of the best fitting distribution
    results : dict
        Dictionary containing detailed test results
    """
    data = np.asarray(data).flatten()
    
    distributions = [
        'norm',      # Normal/Gaussian distribution
        't',         # Student's t-distribution
        'laplace',   # Laplace distribution (double exponential)
        'logistic',  # Logistic distribution
        'cauchy',    # Cauchy distribution
        'gamma',     # Gamma distribution
        'lognorm',   # Log-normal distribution
        'expon',     # Exponential distribution
        'weibull_min', # Weibull distribution
        'beta',      # Beta distribution
        'uniform',   # Uniform distribution
        'chi2',      # Chi-squared distribution
        'f',         # F distribution
        'genextreme' # Generalized extreme value distribution
    ]
    
    results = {}
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        for dist_name in distributions:
            try:
                distribution = getattr(stats, dist_name)
                
                params = distribution.fit(data)
                ks_statistic, ks_pvalue = stats.kstest(data, dist_name, params)
                
                try:
                    ad_result = stats.anderson(data, dist_name)
                    ad_statistic = ad_result.statistic
                    ad_critical_values = ad_result.critical_values
                    ad_pass = ad_statistic < ad_critical_values[2]
                except:
                    ad_statistic = np.nan
                    ad_pass = np.nan
                
                loglik = np.sum(distribution.logpdf(data, *params))
                k = len(params)
                n = len(data)
                aic = 2 * k - 2 * loglik
                bic = k * np.log(n) - 2 * loglik
                
                mean_empirical = np.mean(data)
                var_empirical = np.var(data)
                skew_empirical = stats.skew(data)
                kurt_empirical = stats.kurtosis(data)
                
                try:
                    mean_theoretical = distribution.mean(*params)
                except:
                    mean_theoretical = np.nan
                    
                try:
                    var_theoretical = distribution.var(*params)
                except:
                    var_theoretical = np.nan
                    
                try:
                    skew_theoretical = distribution.stats(*params, moments='s')
                except:
                    skew_theoretical = np.nan
                    
                try:
                    kurt_theoretical = distribution.stats(*params, moments='k')
                except:
                    kurt_theoretical = np.nan
                
                results[dist_name] = {
                    'params': params,
                    'ks_statistic': ks_statistic,
                    'ks_pvalue': ks_pvalue,
                    'ad_statistic': ad_statistic,
                    'aic': aic,
                    'bic': bic,
                    'moment_errors': {
                        'mean': abs(mean_empirical - mean_theoretical) if not np.isnan(mean_theoretical) else np.inf,
                        'var': abs(var_empirical - var_theoretical) if not np.isnan(var_theoretical) else np.inf,
                        'skew': abs(skew_empirical - skew_theoretical) if not np.isnan(skew_theoretical) else np.inf,
                        'kurt': abs(kurt_empirical - kurt_theoretical) if not np.isnan(kurt_theoretical) else np.inf
                    },
                    'pass_ks': ks_pvalue > alpha,
                    'pass_ad': ad_pass
                }
                
            except Exception as e:
                results[dist_name] = {'error': str(e)}
    
    valid_distributions = {k: v for k, v in results.items() if 'error' not in v}
    
    if not valid_distributions:
        return "no_valid_fit", results
    
    for dist_name, result in valid_distributions.items():
        score = 0
        
        if result['pass_ks']:
            score += 10
        
        if result['pass_ad'] is True:
            score += 10
            
        score += result['ks_pvalue'] * 5
        
        valid_aic_values = [d['aic'] for d in valid_distributions.values() 
                          if 'aic' in d and np.isfinite(d['aic'])]
        
        if valid_aic_values:
            max_aic = max(valid_aic_values)
            min_aic = min(valid_aic_values)
            aic_range = max_aic - min_aic
            
            if aic_range > 0 and np.isfinite(result['aic']):
                aic_score = (max_aic - result['aic']) / aic_range
                score += aic_score * 5
        
        moment_error_sum = sum(error for error in result['moment_errors'].values() 
                             if np.isfinite(error))
        
        if moment_error_sum:
            score -= moment_error_sum
        
        result['score'] = score
    
    scored_distributions = {k: v for k, v in valid_distributions.items() 
                         if 'score' in v and np.isfinite(v['score'])}
    
    if not scored_distributions:
        best_dist = max(valid_distributions.items(), 
                      key=lambda x: x[1]['ks_pvalue'] if np.isfinite(x[1]['ks_pvalue']) else -np.inf)[0]
    else:
        best_dist = max(scored_distributions.items(), key=lambda x: x[1]['score'])[0]
    
    if return_all:
        return best_dist, results
    else:
        return best_dist, {best_dist: results[best_dist]}
    
def moments(data: np.ndarray) -> Dict[str, float]:
    """
    Estimate empirical moments of a dataset.

    Parameters:
    -----------
    data : np.ndarray
        1D array of data.

    Returns:
    --------
    Dict[str, float]
        Dictionary containing mean, variance, skewness, and kurtosis.
    """
    data = np.asarray(data).flatten()

    return {
        'mean': np.mean(data),
        'variance': np.var(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data)
    }

def qq_plot(data: np.ndarray, dist_name: str):
    """
    Generate an interactive Q-Q plot using Plotly to visually assess fit to a specified distribution.

    Parameters:
    -----------
    data : np.ndarray
        1D array of data.
    
    dist_name : str
        Name of the distribution to compare to (must be in scipy.stats).
    """
    data = np.asarray(data).flatten()
    
    try:
        dist = getattr(stats, dist_name)
        params = dist.fit(data)

        sorted_data = np.sort(data)
        n = len(sorted_data)
        
        probs = (np.arange(1, n + 1) - 0.5) / n
        theoretical_quants = dist.ppf(probs, *params)

        min_val = min(np.min(theoretical_quants), np.min(sorted_data))
        max_val = max(np.max(theoretical_quants), np.max(sorted_data))

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=theoretical_quants,
            y=sorted_data,
            mode='markers',
            name='Quantiles',
            marker=dict(color='blue', size=6)
        ))

        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='45° Reference Line',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f"Q-Q Plot: Data vs {dist_name}",
            xaxis_title="Theoretical Quantiles",
            yaxis_title="Empirical Quantiles",
            template="plotly_white",
            showlegend=True
        )

        fig.show()

    except Exception as e:
        print(f"Error generating Q-Q plot for {dist_name}: {e}")

def pp_plot(data: np.ndarray, dist_name: str):
    """
    Generate an interactive P-P plot using Plotly to visually assess how well a distribution fits the data.

    Parameters:
    -----------
    data : np.ndarray
        1D array of data.

    dist_name : str
        Name of the distribution to compare to (must exist in scipy.stats).
    """
    data = np.asarray(data).flatten()

    try:
        dist = getattr(stats, dist_name)
        params = dist.fit(data)

        sorted_data = np.sort(data)
        n = len(sorted_data)

        empirical_probs = np.arange(1, n + 1) / (n + 1)
        theoretical_probs = dist.cdf(sorted_data, *params)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=theoretical_probs,
            y=empirical_probs,
            mode='markers',
            name='P-P Points',
            marker=dict(color='green', size=6)
        ))

        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='45° Reference Line',
            line=dict(color='red', dash='dash')
        ))

        fig.update_layout(
            title=f"P-P Plot: Data vs {dist_name}",
            xaxis_title="Theoretical CDF",
            yaxis_title="Empirical CDF",
            template="plotly_white",
            showlegend=True
        )

        fig.show()

    except Exception as e:
        print(f"Error generating P-P plot for {dist_name}: {e}")

def goodness_of_fit_tests(data: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Perform additional goodness-of-fit tests on the data.

    Tests included:
    - Shapiro-Wilk
    - Jarque-Bera
    - D'Agostino's K²

    Parameters:
    -----------
    data : np.ndarray
        1D array of data to test for normality.

    Returns:
    --------
    results : dict
        Dictionary with test statistics and p-values.
    """
    data = np.asarray(data).flatten()
    results = {}

    ## Shapiro-Wilk Test
    try:
        stat, pval = stats.shapiro(data)
        results['Shapiro-Wilk'] = {'statistic': stat, 'p_value': pval}
    except Exception as e:
        results['Shapiro-Wilk'] = {'error': str(e)}

    ## Jarque-Bera Test
    try:
        stat, pval = stats.jarque_bera(data)
        results['Jarque-Bera'] = {'statistic': stat, 'p_value': pval}
    except Exception as e:
        results['Jarque-Bera'] = {'error': str(e)}

    ## D’Agostino’s K² Test
    try:
        stat, pval = stats.normaltest(data)
        results["D'Agostino K²"] = {'statistic': stat, 'p_value': pval}
    except Exception as e:
        results["D'Agostino K²"] = {'error': str(e)}

    return results


def block_maxima_fit(data: np.ndarray, block_size: int = 50) -> Dict[str, object]:
    """
    Fit a Generalized Extreme Value (GEV) distribution using the Block Maxima method,
    using pure NumPy and manual MLE optimization.

    Parameters:
    -----------
    data : np.ndarray
        1D array of observations (e.g., returns, prices).
    block_size : int, default=50
        Number of observations per block to extract maxima from.

    Returns:
    --------
    results : dict
        Dictionary containing GEV parameters, log-likelihood, AIC, and BIC.
    """

    def gev_log_likelihood(params, data):
        c, loc, scale = params
        if scale <= 0:
            return np.inf
        z = (data - loc) / scale
        if c == 0:
            logpdf = -z - np.exp(-z) - np.log(scale)
        else:
            t = 1 + c * z
            if np.any(t <= 0):
                return np.inf
            logpdf = -((1 + 1/c) * np.log(t)) - t**(-1/c) - np.log(scale)
        return -np.sum(logpdf)
    
    data = np.asarray(data).flatten()
    n_blocks = len(data) // block_size

    if n_blocks < 2:
        raise ValueError("Not enough data to form multiple blocks. Increase data size or reduce block_size.")

    blocks = data[:n_blocks * block_size].reshape(n_blocks, block_size)
    block_maxima = blocks.max(axis=1)

    loc_init = np.mean(block_maxima)
    scale_init = np.std(block_maxima)
    shape_init = 0.1

    bounds = [(-1, 1), (None, None), (1e-5, None)]  # Shape, loc, scale bounds
    result = minimize(gev_log_likelihood, x0=[shape_init, loc_init, scale_init],
                      args=(block_maxima,), bounds=bounds)

    if not result.success:
        raise RuntimeError("GEV parameter estimation failed: " + result.message)

    c, loc, scale = result.x
    loglik = -result.fun
    k = 3
    n = len(block_maxima)
    aic = 2 * k - 2 * loglik
    bic = k * np.log(n) - 2 * loglik

    return {
        'params': {'shape': c, 'loc': loc, 'scale': scale},
        'log_likelihood': loglik,
        'aic': aic,
        'bic': bic,
        'n_blocks': n_blocks,
        'block_maxima': block_maxima
    }

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the Kullback-Leibler divergence D_KL(p || q) between two discrete probability distributions.

    Parameters:
    -----------
    p : np.ndarray
        First probability distribution (true distribution).
    q : np.ndarray
        Second probability distribution (approximate distribution).

    Returns:
    --------
    float
        KL divergence value. Returns np.inf if q has zero probability where p > 0.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / np.sum(p)
    q = q / np.sum(q)

    epsilon = 1e-12
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)

    return np.sum(p * np.log(p / q))

def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate the Jensen-Shannon divergence between two discrete probability distributions.

    Parameters:
    -----------
    p : np.ndarray
        First probability distribution.
    q : np.ndarray
        Second probability distribution.

    Returns:
    --------
    float
        Jensen-Shannon divergence (symmetric, bounded between 0 and 1).
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    p = p / np.sum(p)
    q = q / np.sum(q)

    m = 0.5 * (p + q)

    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))