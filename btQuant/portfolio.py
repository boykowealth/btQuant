import numpy as np
from scipy.optimize import minimize

def black_litterman(cov_matrix, pi, P, Q, tau=0.05):
    """
    Black-Litterman model for portfolio optimization.
    
    :param cov_matrix: Covariance matrix of asset returns
    :param pi: Equilibrium (market-implied) returns vector
    :param P: Matrix of views (rows are individual views, columns are assets)
    :param Q: Vector of view returns (the magnitudes of views for each view in P)
    :param tau: Scalar representing uncertainty in the prior (usually 0.05)
    :return: Adjusted expected returns (Black-Litterman posterior)
    """
    tau_cov = tau * cov_matrix
    
    M_inverse = np.linalg.inv(np.linalg.inv(tau_cov) + np.dot(P.T, np.dot(np.linalg.inv(np.dot(P, tau_cov).dot(P.T)), P)))
    adjusted_returns = np.dot(M_inverse, np.dot(np.linalg.inv(tau_cov), pi) + np.dot(P.T, np.linalg.inv(np.dot(P, tau_cov).dot(P.T))).dot(Q))
    
    return adjusted_returns

def mean_variance_optimization(expected_returns, cov_matrix, risk_aversion=0.5, constraints=None):
    """
    Mean-Variance Optimization for portfolio selection.
    
    :param expected_returns: Expected returns for each asset
    :param cov_matrix: Covariance matrix of asset returns
    :param risk_aversion: Risk-aversion parameter (higher value indicates more risk aversion)
    :param constraints: Constraints for optimization (e.g., asset weight bounds, sector constraints)
    :return: Optimal portfolio weights
    """
    num_assets = len(expected_returns)
    initial_weights = np.ones(num_assets) / num_assets 
    
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility
    
    bounds = [(0, 1) for _ in range(num_assets)]
    if constraints:
        ## Additional constraints like sector or liquidity limits can be added here in future updates
        pass
    
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def minimum_variance(cov_matrix):
    """
    Minimum Variance Portfolio optimization.
    
    :param cov_matrix: Covariance matrix of asset returns
    :return: Weights of the Minimum Variance Portfolio
    """
    num_assets = len(cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    def objective(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    bounds = [(0, 1) for _ in range(num_assets)]
    
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def risk_parity(cov_matrix):
    """
    Risk Parity Portfolio optimization.
    
    :param cov_matrix: Covariance matrix of asset returns
    :return: Weights of the Risk Parity Portfolio
    """
    num_assets = len(cov_matrix)
    
    def objective(weights):
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        marginal_risks = np.dot(cov_matrix, weights)
        risk_contributions = weights * marginal_risks / portfolio_variance
        return np.sum((risk_contributions - 1/num_assets) ** 2)
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    bounds = [(0, 1) for _ in range(num_assets)]
    
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def equal_weight(num_assets):
    """
    Equal Weight Portfolio optimization.
    
    :param num_assets: Number of assets in the portfolio
    :return: Weights of the Equal Weight Portfolio
    """
    return np.ones(num_assets) / num_assets

def maximum_diversification(cov_matrix):
    """
    Maximum Diversification Portfolio optimization.
    
    :param cov_matrix: Covariance matrix of asset returns
    :return: Weights of the Maximum Diversification Portfolio
    """
    num_assets = len(cov_matrix)
    
    def objective(weights):
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -np.dot(weights.T, np.dot(cov_matrix, weights)) / portfolio_volatility  # Minimize concentration
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    bounds = [(0, 1) for _ in range(num_assets)]
    
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def lasso(expected_returns, cov_matrix, alpha=0.01):
    """
    LASSO Portfolio optimization using L1 regularization.
    
    :param expected_returns: Expected returns for each asset
    :param cov_matrix: Covariance matrix of asset returns
    :param alpha: Regularization parameter (higher alpha means more shrinkage)
    :return: Weights of the LASSO Portfolio
    """
    num_assets = len(expected_returns)
    
    def objective(weights):
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        lasso_penalty = alpha * np.sum(np.abs(weights))  # L1 penalty term
        return -portfolio_return / portfolio_volatility + lasso_penalty
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    
    bounds = [(0, 1) for _ in range(num_assets)]
    
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x

def tangency_portfolio(expected_returns, cov_matrix, risk_free_rate=0):
    """
    Tangency Portfolio optimization (maximizing the Sharpe ratio).
    
    :param expected_returns: Expected returns for each asset
    :param cov_matrix: Covariance matrix of asset returns
    :param risk_free_rate: Risk-free rate (default is 0)
    :return: Weights of the Tangency Portfolio
    """
    num_assets = len(expected_returns)

    excess_returns = expected_returns - risk_free_rate

    def objective(weights):
        portfolio_return = np.dot(weights, excess_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -portfolio_return / portfolio_volatility
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(num_assets)]
    
    initial_weights = np.ones(num_assets) / num_assets
    result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
    
    return result.x
