import numpy as np

def fama_french3(market_returns, smb, hml, beta_m, beta_smb, beta_hml, risk_free=0.02):
    """
    Calculates expected return using Fama-French 3-factor model.
    
    :param market_returns: Market excess returns (list)
    :param smb: Small-minus-big factor (size effect)
    :param hml: High-minus-low factor (value effect)
    :param beta_m: Beta coefficient for market risk
    :param beta_smb: Beta coefficient for size factor
    :param beta_hml: Beta coefficient for value factor
    :param risk_free: Risk-free rate (default 2%)
    :return: Expected asset return
    """
    excess_market_return = sum(market_returns) / len(market_returns)
    smb_avg = sum(smb) / len(smb)
    hml_avg = sum(hml) / len(hml)
    
    expected_return = risk_free + beta_m * excess_market_return + beta_smb * smb_avg + beta_hml * hml_avg
    return expected_return

def carhart4(market_returns, smb, hml, momentum, beta_m, beta_smb, beta_hml, beta_mom, risk_free=0.02):
    """
    Calculates expected return using Carhart 4-factor model.
    
    :param momentum: Momentum factor (past winners keep winning)
    :return: Expected asset return
    """
    excess_market_return = sum(market_returns) / len(market_returns)
    smb_avg = sum(smb) / len(smb)
    hml_avg = sum(hml) / len(hml)
    momentum_avg = sum(momentum) / len(momentum)

    expected_return = (risk_free + beta_m * excess_market_return + beta_smb * smb_avg +
                       beta_hml * hml_avg + beta_mom * momentum_avg)
    return expected_return

def apt(risk_factors, factor_betas, risk_free=0.02):
    """
    Calculates expected return using Arbitrage Pricing Theory (APT) model.
    
    :param risk_factors: List of factor returns (e.g., GDP growth, inflation)
    :param factor_betas: Corresponding sensitivities (beta values)
    :param risk_free: Risk-free rate
    :return: Expected asset return
    """
    expected_return = risk_free + sum(b * f for b, f in zip(factor_betas, risk_factors))
    return expected_return

def nonlinear(risk_factors, factor_betas, nonlinearity="polynomial", degree=2, risk_free=0.02):
    """
    Calculates expected return using a nonlinear factor model.
    
    :param risk_factors: List of factor returns (e.g., GDP growth, inflation)
    :param factor_betas: Corresponding sensitivities (beta values)
    :param nonlinearity: Type of transformation ("polynomial", "logarithmic")
    :param degree: Polynomial degree (for polynomial models)
    :param risk_free: Risk-free rate
    :return: Expected asset return
    """
    transformed_factors = []
    
    if nonlinearity == "polynomial":
        transformed_factors = [f ** degree for f in risk_factors]
    elif nonlinearity == "logarithmic":
        transformed_factors = [np.log(f + 1) for f in risk_factors]
    
    expected_return = risk_free + sum(b * f for b, f in zip(factor_betas, transformed_factors))
    return expected_return

def liquidity(spread, turnover, beta_spread, beta_turnover, risk_free=0.02):
    """
    Calculates expected return using a liquidity factor model.
    
    :param spread: Bid-ask spread (proxy for liquidity risk)
    :param turnover: Trading volume turnover (proxy for market depth)
    :param beta_spread: Sensitivity to bid-ask spread
    :param beta_turnover: Sensitivity to trading volume
    :param risk_free: Risk-free rate
    :return: Expected asset return
    """
    spread_avg = sum(spread) / len(spread)
    turnover_avg = sum(turnover) / len(turnover)

    expected_return = risk_free + beta_spread * spread_avg + beta_turnover * turnover_avg
    return expected_return

def sentiment(sentiment_scores, beta_sentiment, risk_free=0.02):
    """
    Calculates expected return using a sentiment factor model.
    
    :param sentiment_scores: List of sentiment scores (positive or negative market sentiment)
    :param beta_sentiment: Sensitivity to sentiment factor
    :param risk_free: Risk-free rate
    :return: Expected asset return
    """
    sentiment_avg = sum(sentiment_scores) / len(sentiment_scores)
    
    expected_return = risk_free + beta_sentiment * sentiment_avg
    return expected_return

def volatility(volatility_series, beta_volatility, risk_free=0.02):
    """
    Calculates expected return using a volatility factor model.
    
    :param volatility_series: List of asset volatilities
    :param beta_volatility: Sensitivity to volatility factor
    :param risk_free: Risk-free rate
    :return: Expected asset return
    """
    volatility_avg = sum(volatility_series) / len(volatility_series)
    
    expected_return = risk_free + beta_volatility * volatility_avg
    return expected_return