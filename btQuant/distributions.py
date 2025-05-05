import numpy as np
import scipy.stats as stats
from typing import Tuple, Dict
import warnings

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
    