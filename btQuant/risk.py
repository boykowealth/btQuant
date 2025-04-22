import numpy as np
from scipy.stats import norm, stats

def parametric_var(returns, confidence_level=0.95):
    """
    Parametric Value at Risk (VaR) calculation assuming normal distribution.
    
    :param returns: Series or array of asset returns
    :param confidence_level: The confidence level for VaR calculation (default is 95%)
    :return: VaR at the given confidence level
    """
    mean = np.mean(returns)
    std = np.std(returns)
    
    z_score = norm.ppf(1 - confidence_level)
    
    var = -(mean + z_score * std)
    return var

def historical_var(returns, confidence_level=0.95):
    """
    Historical Simulation Value at Risk (VaR) calculation.
    
    :param returns: Series or array of asset returns
    :param confidence_level: The confidence level for VaR calculation (default is 95%)
    :return: VaR at the given confidence level based on historical simulation
    """
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    var = -sorted_returns[index]
    return var

def parametric_cvar(returns, confidence_level=0.95):
    """
    Parametric Conditional Value at Risk (CVaR) calculation assuming normal distribution.
    
    :param returns: Series or array of asset returns
    :param confidence_level: The confidence level for CVaR calculation (default is 95%)
    :return: CVaR at the given confidence level
    """
    mean = np.mean(returns)
    std = np.std(returns)
    
    z_score = norm.ppf(1 - confidence_level)
    
    cvar = -(mean + std * (norm.pdf(z_score) / (1 - confidence_level) - z_score))
    return cvar

def parametric_cvar(returns, confidence_level=0.95):
    """
    Parametric Conditional Value at Risk (CVaR) calculation assuming normal distribution.
    
    :param returns: Series or array of asset returns
    :param confidence_level: The confidence level for CVaR calculation (default is 95%)
    :return: CVaR at the given confidence level
    """
    mean = np.mean(returns)
    std = np.std(returns)
    
    z_score = norm.ppf(1 - confidence_level)
    
    cvar = -(mean + std * (norm.pdf(z_score) / (1 - confidence_level) - z_score))
    return cvar

def historical_cvar(returns, confidence_level=0.95):
    """
    Historical Simulation Conditional Value at Risk (CVaR) calculation.
    
    :param returns: Series or array of asset returns
    :param confidence_level: The confidence level for CVaR calculation (default is 95%)
    :return: CVaR at the given confidence level based on historical simulation
    """
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    tail_losses = sorted_returns[:index]
    cvar = -np.mean(tail_losses)
    return cvar

def expected_shortfall(returns, confidence_level=0.95):
    """
    Expected Shortfall (ES) calculation. This is the average loss beyond the VaR threshold.
    
    :param returns: Series or array of asset returns
    :param confidence_level: The confidence level for ES calculation (default is 95%)
    :return: ES at the given confidence level
    """
    var = historical_var(returns, confidence_level)
    tail_losses = returns[returns <= -var]
    es = -np.mean(tail_losses)
    return es

def drawdown(returns):
    """
    Calculate the maximum drawdown of a portfolio.
    
    :param returns: Cumulative returns series of the portfolio
    :return: Maximum drawdown (as a fraction of portfolio value)
    """
    cumulative_returns = np.cumsum(returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdowns)
    return max_drawdown

def calmar_ratio(returns, risk_free_rate=0):
    """
    Calculate the Calmar ratio for a portfolio, which is the ratio of the annualized return to the maximum drawdown.
    
    :param returns: Annualized returns series of the portfolio
    :param risk_free_rate: Risk-free rate (optional, default is 0)
    :return: Calmar ratio
    """
    annualized_return = np.mean(returns) * 252  # Assuming daily returns
    max_drawdown = drawdown(returns)
    calmar_ratio = annualized_return / abs(max_drawdown)
    return calmar_ratio

def sharpe_ratio(returns, risk_free_rate=0):
    """
    Calculate the Sharpe Ratio for a series of returns.
    
    :param returns: Array or Series of periodic portfolio returns
    :param risk_free_rate: Risk-free rate (expressed per period, e.g., daily)
    :return: Sharpe Ratio
    """
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / np.std(excess_returns)

def omega_ratio(returns, threshold=0.0):
    """
    Calculate the Omega Ratio of a return series.
    
    :param returns: Array or Series of returns
    :param threshold: Threshold return (e.g., risk-free rate or 0)
    :return: Omega Ratio
    """
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns < threshold]
    
    omega = np.sum(gains) / np.sum(losses) if np.sum(losses) != 0 else np.inf
    return omega

def modified_var(returns, confidence=0.95):
    """
    Computes the modified Value at Risk using the Cornish-Fisher expansion,
    which adjusts for skewness and kurtosis in the return distribution.

    Parameters:
    - returns: array-like, asset or portfolio returns
    - confidence: float, confidence level (e.g. 0.95)

    Returns:
    - Modified Value at Risk estimate
    """
    z = stats.norm.ppf(confidence)
    s = stats.skew(returns)
    k = stats.kurtosis(returns, fisher=False)
    z_cf = z + (1/6)*(z**2 - 1)*s + (1/24)*(z**3 - 3*z)*k - (1/36)*(2*z**3 - 5*z)*s**2
    return -np.mean(returns) + z_cf * np.std(returns)

def hill_tail_index(returns, k=50):
    """
    Estimates the tail index of a return distribution using the Hill estimator,
    useful for detecting heavy (fat) tails.

    Parameters:
    - returns: array-like, asset or portfolio returns
    - k: int, number of top order statistics (extreme values) to use

    Returns:
    - Hill tail index (lower value = heavier tail)
    """
    sorted_returns = -np.sort(-returns)  # descending order
    top_k = sorted_returns[:k]
    return 1 / (np.mean(np.log(top_k / sorted_returns[k])))

def excess_kurtosis_ratio(returns):
    """
    Calculates the ratio of excess kurtosis to the normal distribution's kurtosis,
    to quantify fat-tailed behavior.

    Parameters:
    - returns: array-like

    Returns:
    - Excess kurtosis ratio (Normal = 1)
    """
    return stats.kurtosis(returns) / 3

def copula_crash(returns_x, returns_y, confidence_level=0.95):
    """
    Copula-based crash risk modeling using a Gaussian copula to estimate tail dependence.
    
    :param returns_x: Series or array of asset X returns
    :param returns_y: Series or array of asset Y returns
    :param confidence_level: Confidence level for crash risk estimation (default is 95%)
    :return: Joint probability of extreme losses
    """

    u_x = stats.rankdata(returns_x) / (len(returns_x) + 1)
    u_y = stats.rankdata(returns_y) / (len(returns_y) + 1)
    
    rho = np.corrcoef(returns_x, returns_y)[0, 1]
    
    copula = stats.norm.ppf(u_x), stats.norm.ppf(u_y)
    
    joint_loss_prob = stats.multivariate_normal.cdf(copula, mean=[0, 0], cov=[[1, rho], [rho, 1]])
    
    return joint_loss_prob if joint_loss_prob < 1 - confidence_level else 1 - confidence_level

def copula_gain(returns_x, returns_y, confidence_level=0.95):
    """
    Copula-based right-tail dependence modeling for extreme upward movements.
    
    :param returns_x: Series or array of asset X returns
    :param returns_y: Series or array of asset Y returns
    :param confidence_level: Confidence level for extreme gain estimation (default is 95%)
    :return: Joint probability of extreme gains
    """

    u_x = stats.rankdata(returns_x) / (len(returns_x) + 1)
    u_y = stats.rankdata(returns_y) / (len(returns_y) + 1)
    
    rho = np.corrcoef(returns_x, returns_y)[0, 1]
    theta = 2 * (1 - rho) 
    
    joint_gain_prob = (u_x ** -theta + u_y ** -theta - 1) ** (-1/theta)
    
    return joint_gain_prob if joint_gain_prob > confidence_level else confidence_level

