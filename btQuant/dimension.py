import numpy as np

def pca(X, nComponents=None):
    """
    Principal component analysis.
    
    Parameters:
        X: features (nSamples x nFeatures)
        nComponents: number of components to keep
    
    Returns:
        dict with 'transformed', 'eigenvalues', 'eigenvectors', 'explained_variance'
    """
    XMean = np.mean(X, axis=0)
    XCentered = X - XMean
    
    covMatrix = np.cov(XCentered, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(covMatrix)
    
    sortedIndices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sortedIndices]
    eigenvectors = eigenvectors[:, sortedIndices]
    
    if nComponents is not None:
        eigenvectors = eigenvectors[:, :nComponents]
        eigenvalues = eigenvalues[:nComponents]
    
    XTransformed = XCentered @ eigenvectors
    
    totalVar = np.sum(eigenvalues)
    explainedVar = eigenvalues / totalVar if totalVar > 0 else np.zeros_like(eigenvalues)
    
    return {
        'transformed': XTransformed,
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'explainedVariance': explainedVar,
        'mean': XMean
    }

def lda(X, y, nComponents=None):
    """
    Linear discriminant analysis.
    
    Parameters:
        X: features
        y: class labels
        nComponents: number of components
    
    Returns:
        dict with 'transformed', 'eigenvalues', 'eigenvectors'
    """
    meanOverall = np.mean(X, axis=0)
    
    classes = np.unique(y)
    meanClasses = np.array([np.mean(X[y == c], axis=0) for c in classes])
    
    Sb = np.zeros((X.shape[1], X.shape[1]))
    for c, meanClass in zip(classes, meanClasses):
        nc = X[y == c].shape[0]
        meanDiff = (meanClass - meanOverall).reshape(-1, 1)
        Sb += nc * (meanDiff @ meanDiff.T)
    
    Sw = np.zeros((X.shape[1], X.shape[1]))
    for c, meanClass in zip(classes, meanClasses):
        XC = X[y == c]
        meanDiff = (XC - meanClass).T
        Sw += meanDiff @ meanDiff.T
    
    try:
        SwInv = np.linalg.inv(Sw + 1e-6 * np.eye(Sw.shape[0]))
        eigenvals, eigenvecs = np.linalg.eig(SwInv @ Sb)
    except:
        eigenvals = np.zeros(min(len(classes) - 1, X.shape[1]))
        eigenvecs = np.eye(X.shape[1], len(eigenvals))
    
    sortedIndices = np.argsort(np.abs(eigenvals))[::-1]
    eigenvals = eigenvals[sortedIndices]
    eigenvecs = eigenvecs[:, sortedIndices]
    
    maxComponents = min(len(classes) - 1, X.shape[1])
    if nComponents is not None:
        nComponents = min(nComponents, maxComponents)
    else:
        nComponents = maxComponents
    
    eigenvecs = eigenvecs[:, :nComponents]
    eigenvals = eigenvals[:nComponents]
    
    XTransformed = (X - meanOverall) @ eigenvecs.real
    
    return {
        'transformed': XTransformed,
        'eigenvalues': eigenvals.real,
        'eigenvectors': eigenvecs.real,
        'mean': meanOverall
    }

def tsne(X, nComponents=2, perplexity=30.0, maxIter=1000, learningRate=200.0):
    """
    t-SNE dimensionality reduction.
    
    Parameters:
        X: features
        nComponents: target dimensions
        perplexity: perplexity parameter
        maxIter: maximum iterations
        learningRate: learning rate
    
    Returns:
        dict with 'transformed'
    """
    n = X.shape[0]
    
    distances = np.zeros((n, n))
    for i in range(n):
        distances[i] = np.sqrt(np.sum((X - X[i])**2, axis=1))
    
    P = np.zeros((n, n))
    for i in range(n):
        beta = 1.0
        for _ in range(50):
            Pi = np.exp(-distances[i]**2 * beta)
            Pi[i] = 0
            sumPi = np.sum(Pi)
            if sumPi > 0:
                Pi /= sumPi
            
            H = -np.sum(Pi * np.log(Pi + 1e-10))
            Hdiff = H - np.log(perplexity)
            
            if abs(Hdiff) < 1e-5:
                break
            
            if Hdiff > 0:
                beta *= 2
            else:
                beta /= 2
        
        P[i] = Pi
    
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    
    Y = np.random.randn(n, nComponents) * 0.0001
    
    for iteration in range(maxIter):
        distancesY = np.zeros((n, n))
        for i in range(n):
            distancesY[i] = np.sqrt(np.sum((Y - Y[i])**2, axis=1))
        
        Q = 1 / (1 + distancesY**2)
        np.fill_diagonal(Q, 0)
        Q /= np.sum(Q)
        Q = np.maximum(Q, 1e-12)
        
        grad = np.zeros((n, nComponents))
        for i in range(n):
            diff = Y[i] - Y
            grad[i] = 4 * np.sum(((P[i] - Q[i]) * (1 / (1 + distancesY[i]**2)))[:, np.newaxis] * diff, axis=0)
        
        Y -= learningRate * grad
        
        Y -= np.mean(Y, axis=0)
    
    return {'transformed': Y}

def ica(X, nComponents=None, maxIter=200, tol=1e-4):
    """
    Independent component analysis.
    
    Parameters:
        X: features
        nComponents: number of components
        maxIter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        dict with 'transformed', 'mixingMatrix', 'unmixingMatrix'
    """
    XMean = np.mean(X, axis=0)
    XCentered = X - XMean
    
    covMatrix = np.cov(XCentered, rowvar=False)
    eigenvals, eigenvecs = np.linalg.eigh(covMatrix)
    
    sortedIndices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sortedIndices]
    eigenvecs = eigenvecs[:, sortedIndices]
    
    if nComponents is not None:
        eigenvecs = eigenvecs[:, :nComponents]
        eigenvals = eigenvals[:nComponents]
    else:
        nComponents = X.shape[1]
    
    whitened = XCentered @ eigenvecs @ np.diag(1.0 / np.sqrt(eigenvals + 1e-10))
    
    W = np.random.randn(nComponents, nComponents)
    W /= np.linalg.norm(W, axis=0)
    
    for iteration in range(maxIter):
        WOld = W.copy()
        
        WX = W @ whitened.T
        g = np.tanh(WX)
        gPrime = 1 - g**2
        
        W = (g @ whitened) / whitened.shape[0] - np.diag(np.mean(gPrime, axis=1)) @ W
        
        W = W @ np.linalg.inv(np.sqrt(W @ W.T))
        
        if np.max(np.abs(np.abs(np.diag(W @ WOld.T)) - 1)) < tol:
            break
    
    S = W @ whitened.T
    
    mixingMatrix = np.linalg.pinv(W)
    
    return {
        'transformed': S.T,
        'mixingMatrix': mixingMatrix,
        'unmixingMatrix': W,
        'mean': XMean
    }

def nmf(X, nComponents, maxIter=200, tol=1e-4):
    """
    Non-negative matrix factorization.
    
    Parameters:
        X: non-negative features
        nComponents: number of components
        maxIter: maximum iterations
        tol: convergence tolerance
    
    Returns:
        dict with 'W' (basis), 'H' (coefficients)
    """
    n, m = X.shape
    
    W = np.random.rand(n, nComponents)
    H = np.random.rand(nComponents, m)
    
    for iteration in range(maxIter):
        WOld = W.copy()
        HOld = H.copy()
        
        H = H * (W.T @ X) / (W.T @ W @ H + 1e-10)
        
        W = W * (X @ H.T) / (W @ H @ H.T + 1e-10)
        
        if (np.linalg.norm(W - WOld) + np.linalg.norm(H - HOld)) < tol:
            break
    
    return {'W': W, 'H': H}

def kernelPca(X, nComponents=None, kernel='rbf', gamma=None):
    """
    Kernel PCA for non-linear dimensionality reduction.
    
    Parameters:
        X: features
        nComponents: number of components
        kernel: 'rbf', 'poly', 'linear'
        gamma: kernel parameter
    
    Returns:
        dict with 'transformed', 'eigenvalues', 'eigenvectors'
    """
    n = X.shape[0]
    
    if kernel == 'rbf':
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = np.exp(-gamma * np.sum((X[i] - X[j])**2))
    
    elif kernel == 'poly':
        degree = 3 if gamma is None else int(gamma)
        K = (X @ X.T + 1)**degree
    
    elif kernel == 'linear':
        K = X @ X.T
    
    else:
        raise ValueError("kernel must be 'rbf', 'poly', or 'linear'")
    
    oneN = np.ones((n, n)) / n
    K = K - oneN @ K - K @ oneN + oneN @ K @ oneN
    
    eigenvals, eigenvecs = np.linalg.eigh(K)
    
    sortedIndices = np.argsort(eigenvals)[::-1]
    eigenvals = eigenvals[sortedIndices]
    eigenvecs = eigenvecs[:, sortedIndices]
    
    if nComponents is not None:
        eigenvecs = eigenvecs[:, :nComponents]
        eigenvals = eigenvals[:nComponents]
    
    eigenvals = np.maximum(eigenvals, 0)
    
    XTransformed = eigenvecs * np.sqrt(eigenvals)
    
    return {
        'transformed': XTransformed,
        'eigenvalues': eigenvals,
        'eigenvectors': eigenvecs
    }

def mds(X, nComponents=2, metric=True, maxIter=300):
    """
    Multidimensional scaling.
    
    Parameters:
        X: features or distance matrix
        nComponents: target dimensions
        metric: use metric MDS (True) or non-metric (False)
        maxIter: maximum iterations for non-metric
    
    Returns:
        dict with 'transformed'
    """
    if X.shape[0] != X.shape[1]:
        n = X.shape[0]
        D = np.zeros((n, n))
        for i in range(n):
            D[i] = np.sqrt(np.sum((X - X[i])**2, axis=1))
    else:
        D = X
        n = D.shape[0]
    
    if metric:
        D2 = D**2
        
        H = np.eye(n) - np.ones((n, n)) / n
        
        B = -0.5 * H @ D2 @ H
        
        eigenvals, eigenvecs = np.linalg.eigh(B)
        
        sortedIndices = np.argsort(eigenvals)[::-1]
        eigenvals = eigenvals[sortedIndices]
        eigenvecs = eigenvecs[:, sortedIndices]
        
        eigenvals = np.maximum(eigenvals[:nComponents], 0)
        eigenvecs = eigenvecs[:, :nComponents]
        
        Y = eigenvecs * np.sqrt(eigenvals)
    
    else:
        Y = np.random.randn(n, nComponents)
        
        for iteration in range(maxIter):
            distY = np.zeros((n, n))
            for i in range(n):
                distY[i] = np.sqrt(np.sum((Y - Y[i])**2, axis=1))
            
            stress = np.sum((D - distY)**2)
            
            grad = np.zeros((n, nComponents))
            for i in range(n):
                diff = Y[i] - Y
                grad[i] = -2 * np.sum(((D[i] - distY[i]) / (distY[i] + 1e-10))[:, np.newaxis] * diff, axis=0)
            
            Y -= 0.01 * grad
            
            Y -= np.mean(Y, axis=0)
    
    return {'transformed': Y}

def isomap(X, nComponents=2, nNeighbors=5):
    """
    Isomap non-linear dimensionality reduction.
    
    Parameters:
        X: features
        nComponents: target dimensions
        nNeighbors: number of nearest neighbors
    
    Returns:
        dict with 'transformed'
    """
    n = X.shape[0]
    
    distances = np.zeros((n, n))
    for i in range(n):
        distances[i] = np.sqrt(np.sum((X - X[i])**2, axis=1))
    
    graph = np.full((n, n), np.inf)
    for i in range(n):
        neighbors = np.argsort(distances[i])[1:nNeighbors + 1]
        for j in neighbors:
            graph[i, j] = distances[i, j]
            graph[j, i] = distances[j, i]
    
    geodesic = graph.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if geodesic[i, k] + geodesic[k, j] < geodesic[i, j]:
                    geodesic[i, j] = geodesic[i, k] + geodesic[k, j]
    
    return mds(geodesic, nComponents=nComponents, metric=True)