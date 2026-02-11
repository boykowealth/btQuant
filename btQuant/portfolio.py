import numpy as np

def _minimize(func, x0, bounds=None, maxIter=1000):
    """Simple gradient-free optimization using pattern search."""
    x = np.array(x0, dtype=float)
    n = len(x)
    
    if bounds is None:
        bounds = [(-np.inf, np.inf)] * n
    
    stepSize = 0.1
    minStep = 1e-8
    bestF = func(x)
    
    for iteration in range(maxIter):
        improved = False
        
        for i in range(n):
            xTry = x.copy()
            xTry[i] += stepSize
            xTry[i] = np.clip(xTry[i], bounds[i][0], bounds[i][1])
            fTry = func(xTry)
            
            if fTry < bestF:
                x = xTry
                bestF = fTry
                improved = True
                continue
            
            xTry = x.copy()
            xTry[i] -= stepSize
            xTry[i] = np.clip(xTry[i], bounds[i][0], bounds[i][1])
            fTry = func(xTry)
            
            if fTry < bestF:
                x = xTry
                bestF = fTry
                improved = True
        
        if not improved:
            stepSize *= 0.5
            if stepSize < minStep:
                break
    
    return x

def blackLitterman(covMatrix, pi, P, Q, tau=0.05):
    """
    Black-Litterman model for portfolio optimization.
    
    Parameters:
        covMatrix: covariance matrix of returns
        pi: equilibrium (market-implied) returns
        P: views matrix (rows=views, cols=assets)
        Q: view returns vector
        tau: prior uncertainty (default 0.05)
    
    Returns:
        adjusted expected returns
    """
    tauCov = tau * covMatrix
    
    PtauCovPt = P @ tauCov @ P.T
    PtauCovPtInv = np.linalg.inv(PtauCovPt)
    
    tauCovInv = np.linalg.inv(tauCov)
    
    MInv = np.linalg.inv(tauCovInv + P.T @ PtauCovPtInv @ P)
    adjustedReturns = MInv @ (tauCovInv @ pi + P.T @ PtauCovPtInv @ Q)
    
    return adjustedReturns

def meanVariance(expectedReturns, covMatrix, riskAversion=0.5):
    """
    Mean-variance optimization.
    
    Parameters:
        expectedReturns: expected returns for each asset
        covMatrix: covariance matrix
        riskAversion: risk aversion parameter (higher = more risk averse)
    
    Returns:
        optimal portfolio weights
    """
    nAssets = len(expectedReturns)
    
    def objective(weights):
        portReturn = weights @ expectedReturns
        portVol = np.sqrt(weights @ covMatrix @ weights)
        return -portReturn / portVol
    
    initialWeights = np.ones(nAssets) / nAssets
    bounds = [(0, 1) for _ in range(nAssets)]
    
    bestWeights = _minimize(objective, initialWeights, bounds=bounds)
    
    bestWeights = bestWeights / np.sum(bestWeights)
    
    return bestWeights

def minVariance(covMatrix):
    """
    Minimum variance portfolio.
    
    Parameters:
        covMatrix: covariance matrix
    
    Returns:
        minimum variance portfolio weights
    """
    nAssets = len(covMatrix)
    
    def objective(weights):
        return weights @ covMatrix @ weights
    
    initialWeights = np.ones(nAssets) / nAssets
    bounds = [(0, 1) for _ in range(nAssets)]
    
    bestWeights = _minimize(objective, initialWeights, bounds=bounds)
    
    bestWeights = bestWeights / np.sum(bestWeights)
    
    return bestWeights

def riskParity(covMatrix):
    """
    Risk parity portfolio optimization.
    
    Parameters:
        covMatrix: covariance matrix
    
    Returns:
        risk parity portfolio weights
    """
    nAssets = len(covMatrix)
    
    def objective(weights):
        portVar = weights @ covMatrix @ weights
        marginalRisks = covMatrix @ weights
        riskContribs = weights * marginalRisks / (portVar + 1e-10)
        target = 1.0 / nAssets
        return np.sum((riskContribs - target)**2)
    
    initialWeights = np.ones(nAssets) / nAssets
    bounds = [(0, 1) for _ in range(nAssets)]
    
    bestWeights = _minimize(objective, initialWeights, bounds=bounds)
    
    bestWeights = bestWeights / np.sum(bestWeights)
    
    return bestWeights

def equalWeight(nAssets):
    """
    Equal weight portfolio.
    
    Parameters:
        nAssets: number of assets
    
    Returns:
        equal weight portfolio weights
    """
    return np.ones(nAssets) / nAssets

def maxDiversification(covMatrix):
    """
    Maximum diversification portfolio.
    
    Parameters:
        covMatrix: covariance matrix
    
    Returns:
        maximum diversification portfolio weights
    """
    nAssets = len(covMatrix)
    
    volatilities = np.sqrt(np.diag(covMatrix))
    
    def objective(weights):
        weightedVol = weights @ volatilities
        portVol = np.sqrt(weights @ covMatrix @ weights)
        return -weightedVol / (portVol + 1e-10)
    
    initialWeights = np.ones(nAssets) / nAssets
    bounds = [(0, 1) for _ in range(nAssets)]
    
    bestWeights = _minimize(objective, initialWeights, bounds=bounds)
    
    bestWeights = bestWeights / np.sum(bestWeights)
    
    return bestWeights

def tangency(expectedReturns, covMatrix, riskFreeRate=0):
    """
    Tangency portfolio (maximum Sharpe ratio).
    
    Parameters:
        expectedReturns: expected returns
        covMatrix: covariance matrix
        riskFreeRate: risk-free rate
    
    Returns:
        tangency portfolio weights
    """
    nAssets = len(expectedReturns)
    excessReturns = expectedReturns - riskFreeRate
    
    def objective(weights):
        portReturn = weights @ excessReturns
        portVol = np.sqrt(weights @ covMatrix @ weights)
        return -portReturn / (portVol + 1e-10)
    
    initialWeights = np.ones(nAssets) / nAssets
    bounds = [(0, 1) for _ in range(nAssets)]
    
    bestWeights = _minimize(objective, initialWeights, bounds=bounds)
    
    bestWeights = bestWeights / np.sum(bestWeights)
    
    return bestWeights

def maxSharpe(expectedReturns, covMatrix, riskFreeRate=0):
    """
    Maximum Sharpe ratio portfolio (alias for tangency).
    
    Parameters:
        expectedReturns: expected returns
        covMatrix: covariance matrix
        riskFreeRate: risk-free rate
    
    Returns:
        maximum Sharpe portfolio weights
    """
    return tangency(expectedReturns, covMatrix, riskFreeRate)

def efficientFrontier(expectedReturns, covMatrix, nPoints=50):
    """
    Compute efficient frontier.
    
    Parameters:
        expectedReturns: expected returns
        covMatrix: covariance matrix
        nPoints: number of points on frontier
    
    Returns:
        list of dicts with returns, volatility, weights
    """
    minVar = minVariance(covMatrix)
    minRet = minVar @ expectedReturns
    maxRet = np.max(expectedReturns)
    
    targetReturns = np.linspace(minRet, maxRet, nPoints)
    
    frontier = []
    
    for targetRet in targetReturns:
        nAssets = len(expectedReturns)
        
        def objective(weights):
            return weights @ covMatrix @ weights
        
        initialWeights = np.ones(nAssets) / nAssets
        bounds = [(0, 1) for _ in range(nAssets)]
        
        bestWeights = initialWeights.copy()
        bestVar = objective(bestWeights)
        
        for _ in range(500):
            testWeights = np.random.dirichlet(np.ones(nAssets))
            testRet = testWeights @ expectedReturns
            
            if abs(testRet - targetRet) < 0.01:
                testVar = objective(testWeights)
                if testVar < bestVar:
                    bestWeights = testWeights
                    bestVar = testVar
        
        weights = bestWeights
        portRet = weights @ expectedReturns
        portVol = np.sqrt(weights @ covMatrix @ weights)
        
        frontier.append({
            'return': portRet,
            'volatility': portVol,
            'weights': weights
        })
    
    return frontier

def hierarchicalRiskParity(covMatrix, returns):
    """
    Hierarchical risk parity portfolio allocation.
    
    Parameters:
        covMatrix: covariance matrix
        returns: historical returns (for correlation)
    
    Returns:
        HRP portfolio weights
    """
    nAssets = len(covMatrix)
    
    corr = covMatrix / np.outer(np.sqrt(np.diag(covMatrix)), np.sqrt(np.diag(covMatrix)))
    
    dist = np.sqrt((1 - corr) / 2)
    
    clusters = [[i] for i in range(nAssets)]
    
    while len(clusters) > 1:
        minDist = float('inf')
        mergeI, mergeJ = 0, 1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                clusterDist = 0
                count = 0
                for a in clusters[i]:
                    for b in clusters[j]:
                        clusterDist += dist[a, b]
                        count += 1
                avgDist = clusterDist / count
                
                if avgDist < minDist:
                    minDist = avgDist
                    mergeI, mergeJ = i, j
        
        clusters[mergeI].extend(clusters[mergeJ])
        del clusters[mergeJ]
    
    def allocate(cluster):
        if len(cluster) == 1:
            return {cluster[0]: 1.0}
        
        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]
        
        leftVar = covMatrix[np.ix_(left, left)].sum()
        rightVar = covMatrix[np.ix_(right, right)].sum()
        
        totalVar = leftVar + rightVar
        leftWeight = 1.0 - leftVar / totalVar
        rightWeight = 1.0 - rightVar / totalVar
        
        leftAlloc = allocate(left)
        rightAlloc = allocate(right)
        
        allocation = {}
        for asset, weight in leftAlloc.items():
            allocation[asset] = weight * leftWeight
        for asset, weight in rightAlloc.items():
            allocation[asset] = weight * rightWeight
        
        return allocation
    
    allocation = allocate(clusters[0])
    
    weights = np.zeros(nAssets)
    for asset, weight in allocation.items():
        weights[asset] = weight
    
    weights = weights / np.sum(weights)
    
    return weights

def minCvar(expectedReturns, returns, alpha=0.95):
    """
    Minimum CVaR (Conditional Value at Risk) portfolio.
    
    Parameters:
        expectedReturns: expected returns
        returns: historical returns matrix (nSamples x nAssets)
        alpha: confidence level
    
    Returns:
        minimum CVaR portfolio weights
    """
    nAssets = returns.shape[1]
    
    def cvar(weights):
        portReturns = returns @ weights
        var = np.percentile(portReturns, (1 - alpha) * 100)
        tailLosses = portReturns[portReturns <= var]
        return -np.mean(tailLosses) if len(tailLosses) > 0 else 0
    
    initialWeights = np.ones(nAssets) / nAssets
    bounds = [(0, 1) for _ in range(nAssets)]
    
    bestWeights = _minimize(cvar, initialWeights, bounds=bounds)
    
    bestWeights = bestWeights / np.sum(bestWeights)
    
    return bestWeights

def maxReturn(expectedReturns):
    """
    Maximum return portfolio (100% in highest expected return asset).
    
    Parameters:
        expectedReturns: expected returns
    
    Returns:
        maximum return portfolio weights
    """
    weights = np.zeros(len(expectedReturns))
    weights[np.argmax(expectedReturns)] = 1.0
    return weights