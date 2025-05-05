import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import t, chi2

def ols(y, X, add_constant=True, robust=False, cov_type='HC3'):
    """
    Perform OLS regression with optional robust standard errors.
    
    Parameters:
    -----------
    y : array-like
        Dependent variable
    X : array-like
        Independent variables (predictors)
    add_constant : bool, default=True
        Whether to add a constant term to the model
    robust : bool, default=False
        Whether to use robust standard errors
    cov_type : str, default='HC3'
        Type of robust standard error: 'HC0', 'HC1', 'HC2', 'HC3', or 'HC4'
        
    Returns:
    --------
    dict : Dictionary containing regression results including coefficients, std errors,
           t-stats, p-values, R-squared, adjusted R-squared, and more.
    """
    y = np.asarray(y).flatten()
    X = np.asarray(X)
    
    if add_constant:
        X = np.column_stack([np.ones(len(y)), X])
    
    n, k = X.shape
    
    # β = (X'X)^(-1)X'y
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    
    residuals = y - X @ beta
    y_hat = X @ beta
    
    df = n - k
    
    SSR = sum((y_hat - np.mean(y))**2)
    SST = sum((y - np.mean(y))**2)
    SSE = sum(residuals**2)
    
    r_squared = SSR / SST
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k)
    
    if not robust:
        sigma_squared = SSE / df
        cov_matrix = sigma_squared * XtX_inv
        std_errors = np.sqrt(np.diag(cov_matrix))
    else:
        if cov_type == 'HC0':
            u2 = residuals**2
        elif cov_type == 'HC1':
            u2 = residuals**2 * (n / df)
        elif cov_type == 'HC2':
            h = np.diag(X @ XtX_inv @ X.T)
            u2 = residuals**2 / (1 - h)
        elif cov_type == 'HC3':
            h = np.diag(X @ XtX_inv @ X.T)
            u2 = residuals**2 / (1 - h)**2
        elif cov_type == 'HC4':
            h = np.diag(X @ XtX_inv @ X.T)
            delta = np.minimum(4, h / np.mean(h))
            u2 = residuals**2 / (1 - h)**delta
        else:
            raise ValueError("cov_type must be one of 'HC0', 'HC1', 'HC2', 'HC3', or 'HC4'")
        
        Xu2 = X * u2[:, np.newaxis]
        cov_matrix = XtX_inv @ (X.T @ Xu2) @ XtX_inv
        std_errors = np.sqrt(np.diag(cov_matrix))
    
    t_stats = beta / std_errors
    p_values = 2 * (1 - t.cdf(abs(t_stats), df))
    
    t_crit = t.ppf(0.975, df)
    ci_lower = beta - t_crit * std_errors
    ci_upper = beta + t_crit * std_errors
    
    if add_constant:
        rss1 = SSE
        X_restricted = X[:, 0].reshape(-1, 1)
        beta_restricted = np.linalg.inv(X_restricted.T @ X_restricted) @ X_restricted.T @ y
        residuals_restricted = y - X_restricted @ beta_restricted
        rss0 = sum(residuals_restricted**2)
        
        f_stat = ((rss0 - rss1) / (k - 1)) / (rss1 / df)
        f_p_value = 1 - stats.f.cdf(f_stat, k - 1, df)
    else:
        f_stat = None
        f_p_value = None
    
    return {
        'coefficients': beta,
        'std_errors': std_errors,
        't_stats': t_stats,
        'p_values': p_values,
        'conf_int_lower': ci_lower,
        'conf_int_upper': ci_upper,
        'residuals': residuals,
        'fitted_values': y_hat,
        'r_squared': r_squared,
        'adj_r_squared': adj_r_squared,
        'f_stat': f_stat,
        'f_p_value': f_p_value,
        'n_obs': n,
        'df': df,
        'sse': SSE,
        'cov_matrix': cov_matrix
    }

def white_test(y, X, add_constant=True):
    """
    White's test for heteroskedasticity.
    
    Parameters:
    -----------
    y : array-like
        Dependent variable
    X : array-like
        Independent variables (predictors)
    add_constant : bool, default=True
        Whether to add a constant term to the model
        
    Returns:
    --------
    dict : Dictionary containing test statistic, p-value, and conclusion
    """
    results = ols(y, X, add_constant=add_constant)
    resid = results['residuals']
    
    # Square the residuals
    resid_sq = resid**2
    
    # Create matrix of explanatory variables and their squares/cross-products
    X_orig = np.asarray(X)
    if add_constant:
        X_orig = X_orig if X_orig.ndim > 1 else X_orig.reshape(-1, 1)
    else:
        X_orig = X_orig if X_orig.ndim > 1 else X_orig.reshape(-1, 1)
    
    n, k = X_orig.shape
    
    X_white = []
    
    for i in range(k):
        X_white.append(X_orig[:, i])
    
    for i in range(k):
        X_white.append(X_orig[:, i]**2)
    
    for i in range(k):
        for j in range(i+1, k):
            X_white.append(X_orig[:, i] * X_orig[:, j])
    
    X_white = np.column_stack(X_white)
    
    aux_results = ols(resid_sq, X_white, add_constant=True)
    
    test_stat = n * aux_results['r_squared']
    df = X_white.shape[1]
    p_value = 1 - chi2.cdf(test_stat, df)
    
    conclusion = "Reject null hypothesis of homoskedasticity" if p_value < 0.05 else "Fail to reject null hypothesis of homoskedasticity"
    
    return {
        'test_statistic': test_stat,
        'p_value': p_value,
        'df': df,
        'conclusion': conclusion
    }

def breusch_pagan_test(y, X, add_constant=True):
    """
    Breusch-Pagan test for heteroskedasticity.
    
    Parameters:
    -----------
    y : array-like
        Dependent variable
    X : array-like
        Independent variables (predictors)
    add_constant : bool, default=True
        Whether to add a constant term to the model
        
    Returns:
    --------
    dict : Dictionary containing test statistic, p-value, and conclusion
    """
    results = ols(y, X, add_constant=add_constant)
    resid = results['residuals']
    
    n = len(resid)
    sigma2 = np.sum(resid**2) / n
    resid_sq_norm = resid**2 / sigma2

    bp_results = ols(resid_sq_norm, X, add_constant=add_constant)

    explained_ss = bp_results['r_squared'] * np.sum((resid_sq_norm - np.mean(resid_sq_norm))**2)
    test_stat = 0.5 * explained_ss

    df = X.shape[1] if X.ndim > 1 else 1
    p_value = 1 - chi2.cdf(test_stat, df)
    
    conclusion = "Reject null hypothesis of homoskedasticity" if p_value < 0.05 else "Fail to reject null hypothesis of homoskedasticity"
    
    return {
        'test_statistic': test_stat,
        'p_value': p_value,
        'df': df,
        'conclusion': conclusion
    }

def durbin_watson(residuals):
    """
    Calculate the Durbin-Watson statistic for autocorrelation.
    
    Parameters:
    -----------
    residuals : array-like
        Residuals from a regression model
        
    Returns:
    --------
    float : Durbin-Watson statistic
    """
    residuals = np.asarray(residuals)
    n = len(residuals)
    
    diff_squared = np.sum(np.diff(residuals)**2)
    resid_squared = np.sum(residuals**2)
    
    dw_stat = diff_squared / resid_squared
    
    # Interpretation:
    # DW = 2: No autocorrelation
    # 0 < DW < 2: Positive autocorrelation
    # 2 < DW < 4: Negative autocorrelation
    
    return dw_stat

def ljung_box_test(residuals, lags=None, return_df=False):
    """
    Ljung-Box test for autocorrelation in residuals.
    
    Parameters:
    -----------
    residuals : array-like
        Residuals from a regression model
    lags : int, default=None
        Number of lags to include in the test. If None, min(10, n//5) is used.
    return_df : bool, default=False
        Whether to return a DataFrame with results for each lag
        
    Returns:
    --------
    dict or DataFrame : Test statistics, p-values, and critical values
    """
    residuals = np.asarray(residuals)
    n = len(residuals)
    
    if lags is None:
        lags = min(10, n // 5)
    
    acf_values = []
    for k in range(1, lags + 1):
        numerator = np.sum((residuals[k:] * residuals[:-k]))
        denominator = np.sum(residuals**2)
        acf_values.append(numerator / denominator)
    
    acf_values = np.array(acf_values)
    
    q_stats = []
    p_values = []
    
    for l in range(1, lags + 1):
        q = n * (n + 2) * np.sum((acf_values[:l]**2) / (n - np.arange(1, l + 1)))
        q_stats.append(q)
        p_values.append(1 - chi2.cdf(q, l))
    
    if return_df:
        return pd.DataFrame({
            'lag': range(1, lags + 1),
            'autocorrelation': acf_values,
            'q_stat': q_stats,
            'p_value': p_values,
            'critical_value': chi2.ppf(0.95, np.arange(1, lags + 1))
        })
    else:
        return {
            'lags': range(1, lags + 1),
            'autocorrelations': acf_values,
            'q_stats': q_stats,
            'p_values': p_values,
            'critical_values': chi2.ppf(0.95, np.arange(1, lags + 1))
        }

# Unit Root Tests
def adf_test(series, lags=None, regression='c', autolag='AIC'):
    """
    Augmented Dickey-Fuller test for unit root.
    
    Parameters:
    -----------
    series : array-like
        Time series to test for unit root
    lags : int, default=None
        Number of lags to include in the test. If None, it's determined by autolag.
    regression : str, default='c'
        Regression type: 'nc' for no constant, 'c' for constant, 'ct' for constant and trend
    autolag : str, default='AIC'
        Method to determine the lag length: 'AIC' or 'BIC'
        
    Returns:
    --------
    dict : Dictionary containing test statistic, p-value, critical values, and conclusion
    """
    series = np.asarray(series)
    n = len(series)
    
    if lags is None:
        max_lags = int(np.ceil(12 * (n / 100)**(1/4)))
    else:
        max_lags = lags
    
    if regression == 'nc':
        const = np.zeros(n - 1)
        trend = np.zeros(n - 1)
    elif regression == 'c':
        const = np.ones(n - 1)
        trend = np.zeros(n - 1)
    elif regression == 'ct':
        const = np.ones(n - 1)
        trend = np.arange(1, n)
    else:
        raise ValueError("regression must be one of 'nc', 'c', or 'ct'")
    
    dy = np.diff(series)
    y_1 = series[:-1]
    
    if lags is None and autolag is not None:
        results = {}
        for p in range(max_lags + 1):
            X = []
            if regression != 'nc':
                X.append(const)
            if regression == 'ct':
                X.append(trend)
            X.append(y_1)
            
            for i in range(1, p + 1):
                lag = np.zeros(n - 1)
                lag[i:] = dy[:-i]
                X.append(lag)
            
            if not X:
                X = np.zeros((n - 1, 0))
            else:
                X = np.column_stack(X)
            
            model = ols(dy, X, add_constant=False)
            
            k = X.shape[1]
            ssr = np.sum(model['residuals']**2)
            
            if autolag == 'AIC':
                ic = np.log(ssr / (n - 1)) + 2 * k / (n - 1)
            elif autolag == 'BIC':
                ic = np.log(ssr / (n - 1)) + k * np.log(n - 1) / (n - 1)
            else:
                raise ValueError("autolag must be one of 'AIC' or 'BIC'")
            
            results[p] = {
                'aic': ic,
                'model': model,
                'X': X
            }
        
        optimal_lag = min(results.keys(), key=lambda x: results[x]['aic'])
        model = results[optimal_lag]['model']
        X = results[optimal_lag]['X']
    else:
        p = max_lags
        X = []
        if regression != 'nc':
            X.append(const)
        if regression == 'ct':
            X.append(trend)
        X.append(y_1)
        
        for i in range(1, p + 1):
            lag = np.zeros(n - 1)
            lag[i:] = dy[:-i]
            X.append(lag)
        
        if not X:
            X = np.zeros((n - 1, 0))
        else:
            X = np.column_stack(X)
        
        model = ols(dy, X, add_constant=False)
    
    idx = 0
    if regression != 'nc':
        idx += 1
    if regression == 'ct':
        idx += 1
    
    adf_stat = model['t_stats'][idx]
    
    if regression == 'nc':
        crit_vals = {
            '1%': -2.58,
            '5%': -1.95,
            '10%': -1.62
        }
    elif regression == 'c':
        crit_vals = {
            '1%': -3.43,
            '5%': -2.86,
            '10%': -2.57
        }
    elif regression == 'ct':
        crit_vals = {
            '1%': -3.96,
            '5%': -3.41,
            '10%': -3.13
        }
    
    if regression == 'nc':
        p_value = stats.norm.sf(adf_stat)
    else:
        tau = abs(adf_stat)
        if regression == 'c':
            p_value = np.exp(-0.5 * tau) if tau < 4.38 else 0.01
        else: 
            p_value = np.exp(-0.5 * tau) if tau < 4.65 else 0.01
    
    conclusion = "Reject null hypothesis of a unit root" if adf_stat < crit_vals['5%'] else "Fail to reject null hypothesis of a unit root"
    
    return {
        'test_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': crit_vals,
        'conclusion': conclusion,
        'optimal_lag': optimal_lag if lags is None else max_lags
    }

def kpss_test(series, lags=None, regression='c'):
    """
    KPSS test for stationarity.
    
    Parameters:
    -----------
    series : array-like
        Time series to test for stationarity
    lags : int, default=None
        Number of lags to include in the test. If None, lags = int(12 * (n/100)^0.25)
    regression : str, default='c'
        Regression type: 'c' for constant, 'ct' for constant and trend
        
    Returns:
    --------
    dict : Dictionary containing test statistic, p-value, critical values, and conclusion
    """
    series = np.asarray(series)
    n = len(series)
    
    if lags is None:
        lags = int(np.ceil(12 * (n / 100)**(1/4)))
    
    if regression == 'c':
        resid = series - np.mean(series)
    elif regression == 'ct':
        t = np.arange(1, n + 1)
        X = np.column_stack([np.ones(n), t])
        beta = np.linalg.lstsq(X, series, rcond=None)[0]
        resid = series - X @ beta
    else:
        raise ValueError("regression must be one of 'c' or 'ct'")
    
    s = np.cumsum(resid)
    
    gamma0 = np.sum(resid**2) / n
    
    auto_cov = np.zeros(lags + 1)
    auto_cov[0] = gamma0
    
    for l in range(1, lags + 1):
        auto_cov[l] = np.sum(resid[l:] * resid[:-l]) / n
    
    w = 1 - np.arange(1, lags + 1) / (lags + 1)
    
    s2 = gamma0 + 2 * np.sum(w * auto_cov[1:])
    
    kpss_stat = np.sum(s**2) / (n**2 * s2)
    
    if regression == 'c':
        crit_vals = {
            '1%': 0.739,
            '5%': 0.463,
            '10%': 0.347
        }
    else: 
        crit_vals = {
            '1%': 0.216,
            '5%': 0.146,
            '10%': 0.119
        }
    
    if regression == 'c':
        p_value = np.exp(-0.5 * kpss_stat) if kpss_stat < 2.0 else 0.01
    else: 
        p_value = np.exp(-0.5 * kpss_stat) if kpss_stat < 1.0 else 0.01
    
    conclusion = "Reject null hypothesis of stationarity" if kpss_stat > crit_vals['5%'] else "Fail to reject null hypothesis of stationarity"
    
    return {
        'test_statistic': kpss_stat,
        'p_value': p_value,
        'critical_values': crit_vals,
        'conclusion': conclusion,
        'lags': lags
    }
