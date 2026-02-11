import numpy as np

def famaFrench3(marketReturns, smb, hml, betaM, betaSmb, betaHml, riskFree=0.02):
    """
    Fama-French 3-factor model expected return.
    
    Parameters:
        marketReturns: market excess returns (array)
        smb: size factor returns (array)
        hml: value factor returns (array)
        betaM: market beta
        betaSmb: size beta
        betaHml: value beta
        riskFree: risk-free rate
    
    Returns:
        expected return
    """
    marketReturns = np.asarray(marketReturns)
    smb = np.asarray(smb)
    hml = np.asarray(hml)
    
    excessMarket = np.mean(marketReturns)
    smbAvg = np.mean(smb)
    hmlAvg = np.mean(hml)
    
    expectedReturn = riskFree + betaM * excessMarket + betaSmb * smbAvg + betaHml * hmlAvg
    
    return expectedReturn

def carhart4(marketReturns, smb, hml, momentum, betaM, betaSmb, betaHml, betaMom, riskFree=0.02):
    """
    Carhart 4-factor model expected return.
    
    Parameters:
        marketReturns: market excess returns
        smb: size factor
        hml: value factor
        momentum: momentum factor
        betaM: market beta
        betaSmb: size beta
        betaHml: value beta
        betaMom: momentum beta
        riskFree: risk-free rate
    
    Returns:
        expected return
    """
    marketReturns = np.asarray(marketReturns)
    smb = np.asarray(smb)
    hml = np.asarray(hml)
    momentum = np.asarray(momentum)
    
    excessMarket = np.mean(marketReturns)
    smbAvg = np.mean(smb)
    hmlAvg = np.mean(hml)
    momAvg = np.mean(momentum)
    
    expectedReturn = (riskFree + betaM * excessMarket + betaSmb * smbAvg + 
                     betaHml * hmlAvg + betaMom * momAvg)
    
    return expectedReturn

def apt(riskFactors, factorBetas, riskFree=0.02):
    """
    Arbitrage Pricing Theory expected return.
    
    Parameters:
        riskFactors: array of factor returns
        factorBetas: array of factor sensitivities
        riskFree: risk-free rate
    
    Returns:
        expected return
    """
    riskFactors = np.asarray(riskFactors)
    factorBetas = np.asarray(factorBetas)
    
    expectedReturn = riskFree + np.sum(factorBetas * riskFactors)
    
    return expectedReturn

def capm(marketReturn, beta, riskFree=0.02):
    """
    Capital Asset Pricing Model expected return.
    
    Parameters:
        marketReturn: market return
        beta: systematic risk
        riskFree: risk-free rate
    
    Returns:
        expected return
    """
    return riskFree + beta * (marketReturn - riskFree)

def estimateBeta(assetReturns, marketReturns):
    """
    Estimate beta coefficient.
    
    Parameters:
        assetReturns: asset returns
        marketReturns: market returns
    
    Returns:
        dict with beta, alpha, rSquared
    """
    assetReturns = np.asarray(assetReturns)
    marketReturns = np.asarray(marketReturns)
    
    covMatrix = np.cov(assetReturns, marketReturns)
    beta = covMatrix[0, 1] / covMatrix[1, 1]
    
    alpha = np.mean(assetReturns) - beta * np.mean(marketReturns)
    
    yHat = alpha + beta * marketReturns
    ssRes = np.sum((assetReturns - yHat)**2)
    ssTot = np.sum((assetReturns - np.mean(assetReturns))**2)
    rSquared = 1 - ssRes / ssTot
    
    return {'beta': beta, 'alpha': alpha, 'rSquared': rSquared}

def estimateFactorLoading(assetReturns, factorReturns):
    """
    Estimate factor loadings via OLS.
    
    Parameters:
        assetReturns: asset returns (1D array)
        factorReturns: factor returns (2D array, nObs x nFactors)
    
    Returns:
        dict with loadings, intercept, rSquared
    """
    assetReturns = np.asarray(assetReturns)
    factorReturns = np.asarray(factorReturns)
    
    if factorReturns.ndim == 1:
        factorReturns = factorReturns.reshape(-1, 1)
    
    X = np.column_stack([np.ones(len(assetReturns)), factorReturns])
    
    XtX = X.T @ X
    XtY = X.T @ assetReturns
    
    params = np.linalg.solve(XtX, XtY)
    
    intercept = params[0]
    loadings = params[1:]
    
    yHat = X @ params
    ssRes = np.sum((assetReturns - yHat)**2)
    ssTot = np.sum((assetReturns - np.mean(assetReturns))**2)
    rSquared = 1 - ssRes / ssTot
    
    return {'loadings': loadings, 'intercept': intercept, 'rSquared': rSquared}

def rollingBeta(assetReturns, marketReturns, window=60):
    """
    Calculate rolling beta.
    
    Parameters:
        assetReturns: asset returns
        marketReturns: market returns
        window: rolling window size
    
    Returns:
        array of rolling betas
    """
    assetReturns = np.asarray(assetReturns)
    marketReturns = np.asarray(marketReturns)
    
    n = len(assetReturns)
    betas = np.zeros(n - window + 1)
    
    for i in range(n - window + 1):
        assetWindow = assetReturns[i:i + window]
        marketWindow = marketReturns[i:i + window]
        
        cov = np.cov(assetWindow, marketWindow)[0, 1]
        var = np.var(marketWindow, ddof=1)
        
        betas[i] = cov / var if var > 0 else 0
    
    return betas

def pcaFactors(returns, nFactors=3):
    """
    Extract principal component factors.
    
    Parameters:
        returns: returns matrix (nObs x nAssets)
        nFactors: number of factors to extract
    
    Returns:
        dict with factors, loadings, explainedVariance
    """
    returns = np.asarray(returns)
    
    meanReturns = np.mean(returns, axis=0)
    centeredReturns = returns - meanReturns
    
    covMatrix = np.cov(centeredReturns, rowvar=False)
    
    eigenvals, eigenvecs = np.linalg.eigh(covMatrix)
    
    sortedIndices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sortedIndices]
    eigenvecs = eigenvecs[:, sortedIndices]
    
    eigenvals = eigenvals[:nFactors]
    eigenvecs = eigenvecs[:, :nFactors]
    
    factors = centeredReturns @ eigenvecs
    
    totalVar = np.sum(eigenvals)
    explainedVar = eigenvals / totalVar if totalVar > 0 else np.zeros(nFactors)
    
    return {
        'factors': factors,
        'loadings': eigenvecs,
        'explainedVariance': explainedVar
    }

def factorMimicking(assetReturns, characteristicData, nPortfolios=5):
    """
    Create factor-mimicking portfolios (e.g., SMB, HML).
    
    Parameters:
        assetReturns: returns matrix (nObs x nAssets)
        characteristicData: characteristic values (1D array, nAssets)
        nPortfolios: number of portfolios for sorting
    
    Returns:
        dict with longShort factor returns
    """
    assetReturns = np.asarray(assetReturns)
    characteristicData = np.asarray(characteristicData)
    
    nObs = assetReturns.shape[0]
    
    sortedIndices = np.argsort(characteristicData)
    nPerPortfolio = len(sortedIndices) // nPortfolios
    
    highIndices = sortedIndices[-nPerPortfolio:]
    lowIndices = sortedIndices[:nPerPortfolio]
    
    highReturns = np.mean(assetReturns[:, highIndices], axis=1)
    lowReturns = np.mean(assetReturns[:, lowIndices], axis=1)
    
    factorReturns = highReturns - lowReturns
    
    return {'factorReturns': factorReturns}

def jensenAlpha(assetReturns, marketReturns, riskFree=0.0):
    """
    Calculate Jensen's alpha.
    
    Parameters:
        assetReturns: asset returns
        marketReturns: market returns
        riskFree: risk-free rate per period
    
    Returns:
        Jensen's alpha
    """
    assetReturns = np.asarray(assetReturns)
    marketReturns = np.asarray(marketReturns)
    
    assetExcess = assetReturns - riskFree
    marketExcess = marketReturns - riskFree
    
    beta = np.cov(assetExcess, marketExcess)[0, 1] / np.var(marketExcess, ddof=1)
    
    alpha = np.mean(assetExcess) - beta * np.mean(marketExcess)
    
    return alpha

def treynorMazuy(assetReturns, marketReturns, riskFree=0.0):
    """
    Treynor-Mazuy market timing model.
    
    Parameters:
        assetReturns: asset returns
        marketReturns: market returns
        riskFree: risk-free rate
    
    Returns:
        dict with alpha, beta, gamma (timing coefficient)
    """
    assetReturns = np.asarray(assetReturns)
    marketReturns = np.asarray(marketReturns)
    
    assetExcess = assetReturns - riskFree
    marketExcess = marketReturns - riskFree
    
    marketExcess2 = marketExcess**2
    
    X = np.column_stack([np.ones(len(assetExcess)), marketExcess, marketExcess2])
    
    params = np.linalg.lstsq(X, assetExcess, rcond=None)[0]
    
    return {'alpha': params[0], 'beta': params[1], 'gamma': params[2]}

def informationRatio(assetReturns, benchmarkReturns):
    """
    Information ratio (active return / tracking error).
    
    Parameters:
        assetReturns: portfolio returns
        benchmarkReturns: benchmark returns
    
    Returns:
        information ratio
    """
    assetReturns = np.asarray(assetReturns)
    benchmarkReturns = np.asarray(benchmarkReturns)
    
    activeReturns = assetReturns - benchmarkReturns
    
    return np.mean(activeReturns) / (np.std(activeReturns, ddof=1) + 1e-10)

def appraisalRatio(alpha, residualRisk):
    """
    Appraisal ratio (alpha / residual risk).
    
    Parameters:
        alpha: Jensen's alpha
        residualRisk: residual standard deviation
    
    Returns:
        appraisal ratio
    """
    return alpha / (residualRisk + 1e-10)

def trackingError(assetReturns, benchmarkReturns):
    """
    Tracking error (std dev of active returns).
    
    Parameters:
        assetReturns: portfolio returns
        benchmarkReturns: benchmark returns
    
    Returns:
        tracking error
    """
    assetReturns = np.asarray(assetReturns)
    benchmarkReturns = np.asarray(benchmarkReturns)
    
    activeReturns = assetReturns - benchmarkReturns
    
    return np.std(activeReturns, ddof=1)

def multifactor(assetReturns, factorReturns, riskFree=0.0):
    """
    Multi-factor model regression.
    
    Parameters:
        assetReturns: asset returns
        factorReturns: matrix of factor returns (nObs x nFactors)
        riskFree: risk-free rate
    
    Returns:
        dict with alpha, betas, rSquared, residuals
    """
    assetReturns = np.asarray(assetReturns)
    factorReturns = np.asarray(factorReturns)
    
    if factorReturns.ndim == 1:
        factorReturns = factorReturns.reshape(-1, 1)
    
    assetExcess = assetReturns - riskFree
    
    X = np.column_stack([np.ones(len(assetExcess)), factorReturns])
    
    params = np.linalg.lstsq(X, assetExcess, rcond=None)[0]
    
    alpha = params[0]
    betas = params[1:]
    
    yHat = X @ params
    residuals = assetExcess - yHat
    
    ssRes = np.sum(residuals**2)
    ssTot = np.sum((assetExcess - np.mean(assetExcess))**2)
    rSquared = 1 - ssRes / ssTot
    
    return {
        'alpha': alpha,
        'betas': betas,
        'rSquared': rSquared,
        'residuals': residuals
    }