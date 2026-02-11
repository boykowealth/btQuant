import numpy as np

def regressionTree(X, y, maxDepth=5):
    """
    Regression tree using MSE splits.
    
    Parameters:
        X: features (n_samples, n_features)
        y: target values
        maxDepth: maximum tree depth
    
    Returns:
        dict: tree structure
    """
    def mse(y):
        return np.mean((y - np.mean(y))**2)
    
    def buildTree(X, y, depth):
        if depth == maxDepth or len(np.unique(y)) == 1 or len(y) < 2:
            return np.mean(y)
        
        bestMse = float('inf')
        bestSplit = None
        bestLeftX, bestRightX = None, None
        bestLeftY, bestRightY = None, None
        
        nFeatures = X.shape[1]
        for feature in range(nFeatures):
            uniqueVals = np.unique(X[:, feature])
            for value in uniqueVals:
                leftMask = X[:, feature] <= value
                rightMask = ~leftMask
                
                if np.sum(leftMask) == 0 or np.sum(rightMask) == 0:
                    continue
                
                leftX, rightX = X[leftMask], X[rightMask]
                leftY, rightY = y[leftMask], y[rightMask]
                
                splitMse = (mse(leftY) * len(leftY) + mse(rightY) * len(rightY)) / len(y)
                
                if splitMse < bestMse:
                    bestMse = splitMse
                    bestSplit = (feature, value)
                    bestLeftX, bestRightX = leftX, rightX
                    bestLeftY, bestRightY = leftY, rightY
        
        if bestSplit is None:
            return np.mean(y)
        
        feature, value = bestSplit
        leftTree = buildTree(bestLeftX, bestLeftY, depth + 1)
        rightTree = buildTree(bestRightX, bestRightY, depth + 1)
        
        return {'feature': feature, 'value': value, 'left': leftTree, 'right': rightTree}
    
    return buildTree(X, y, 0)

def predictTree(tree, X):
    """
    Predict using a regression or decision tree.
    
    Parameters:
        tree: tree structure from regressionTree or decisionTree
        X: features to predict (n_samples, n_features)
    
    Returns:
        predictions array
    """
    def predictSingle(tree, x):
        while isinstance(tree, dict):
            if x[tree['feature']] <= tree['value']:
                tree = tree['left']
            else:
                tree = tree['right']
        return tree
    
    return np.array([predictSingle(tree, x) for x in X])

def isolationForest(X, nTrees=100, maxSamples=None, maxDepth=10):
    """
    Isolation forest for anomaly detection.
    
    Parameters:
        X: features (n_samples, n_features)
        nTrees: number of trees
        maxSamples: samples per tree (default all)
        maxDepth: maximum tree depth
    
    Returns:
        list of isolation trees
    """
    def buildTree(X, depth):
        if depth >= maxDepth or len(X) <= 1:
            return {'size': len(X)}
        
        feature = np.random.randint(0, X.shape[1])
        minVal, maxVal = X[:, feature].min(), X[:, feature].max()
        
        if minVal == maxVal:
            return {'size': len(X)}
        
        splitVal = np.random.uniform(minVal, maxVal)
        
        leftMask = X[:, feature] <= splitVal
        rightMask = ~leftMask
        
        if np.sum(leftMask) == 0 or np.sum(rightMask) == 0:
            return {'size': len(X)}
        
        leftTree = buildTree(X[leftMask], depth + 1)
        rightTree = buildTree(X[rightMask], depth + 1)
        
        return {'feature': feature, 'value': splitVal, 'left': leftTree, 'right': rightTree}
    
    trees = []
    nSamples = X.shape[0]
    if maxSamples is None:
        maxSamples = min(256, nSamples)
    
    for _ in range(nTrees):
        indices = np.random.choice(nSamples, min(maxSamples, nSamples), replace=False)
        tree = buildTree(X[indices], 0)
        trees.append(tree)
    
    return trees

def anomalyScore(trees, X):
    """
    Compute anomaly scores from isolation forest.
    
    Parameters:
        trees: list of isolation trees
        X: features to score
    
    Returns:
        anomaly scores (higher = more anomalous)
    """
    def pathLength(tree, x, currentDepth=0):
        if not isinstance(tree, dict) or 'size' in tree:
            size = tree.get('size', 1)
            if size <= 1:
                return currentDepth
            c = 2 * (np.log(size - 1) + 0.5772156649) - 2 * (size - 1) / size
            return currentDepth + c
        
        if x[tree['feature']] <= tree['value']:
            return pathLength(tree['left'], x, currentDepth + 1)
        else:
            return pathLength(tree['right'], x, currentDepth + 1)
    
    n = len(trees[0]) if isinstance(trees[0], dict) else 256
    c = 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n
    
    scores = []
    for x in X:
        avgPath = np.mean([pathLength(tree, x) for tree in trees])
        score = 2 ** (-avgPath / c)
        scores.append(score)
    
    return np.array(scores)

def kmeans(X, k=3, maxIters=100, tol=1e-4):
    """
    K-means clustering.
    
    Parameters:
        X: features (n_samples, n_features)
        k: number of clusters
        maxIters: maximum iterations
        tol: convergence tolerance
    
    Returns:
        centroids, labels
    """
    nSamples = X.shape[0]
    centroids = X[np.random.choice(nSamples, k, replace=False)]
    
    for iteration in range(maxIters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        newCentroids = np.array([X[labels == i].mean(axis=0) if np.sum(labels == i) > 0 
                                 else centroids[i] for i in range(k)])
        
        if np.linalg.norm(centroids - newCentroids) < tol:
            break
        
        centroids = newCentroids
    
    return centroids, labels

def knn(XTrain, yTrain, XTest, k=3):
    """
    K-nearest neighbors classifier.
    
    Parameters:
        XTrain: training features
        yTrain: training labels
        XTest: test features
        k: number of neighbors
    
    Returns:
        predicted labels
    """
    predictions = []
    
    for testPoint in XTest:
        distances = np.sqrt(np.sum((XTrain - testPoint)**2, axis=1))
        sortedIndices = np.argsort(distances)
        nearestNeighbors = yTrain[sortedIndices[:k]]
        mostCommon = np.bincount(nearestNeighbors.astype(int)).argmax()
        predictions.append(mostCommon)
    
    return np.array(predictions)

def naiveBayes(XTrain, yTrain, XTest):
    """
    Gaussian naive Bayes classifier.
    
    Parameters:
        XTrain: training features
        yTrain: training labels
        XTest: test features
    
    Returns:
        predicted labels
    """
    def gaussianPdf(x, mean, std):
        eps = 1e-10
        return (1 / (std + eps) / np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / (std + eps))**2)
    
    classes = np.unique(yTrain)
    meanStd = {}
    
    for cls in classes:
        XCls = XTrain[yTrain == cls]
        meanStd[cls] = {
            'mean': np.mean(XCls, axis=0),
            'std': np.std(XCls, axis=0) + 1e-10
        }
    
    predictions = []
    
    for testPoint in XTest:
        classProbs = {}
        
        for cls in classes:
            likelihood = np.prod(gaussianPdf(testPoint, meanStd[cls]['mean'], meanStd[cls]['std']))
            classProbs[cls] = likelihood
        
        predictedClass = max(classProbs, key=classProbs.get)
        predictions.append(predictedClass)
    
    return np.array(predictions)

def decisionTree(X, y, maxDepth=5):
    """
    Decision tree classifier using Gini impurity.
    
    Parameters:
        X: features (n_samples, n_features)
        y: target labels
        maxDepth: maximum tree depth
    
    Returns:
        dict: tree structure
    """
    def giniImpurity(y):
        if len(y) == 0:
            return 0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)
    
    def bestSplit(X, y):
        bestGini = float('inf')
        bestSplit = None
        bestLeftX, bestRightX = None, None
        bestLeftY, bestRightY = None, None
        
        for feature in range(X.shape[1]):
            uniqueVals = np.unique(X[:, feature])
            for value in uniqueVals:
                leftMask = X[:, feature] <= value
                rightMask = ~leftMask
                
                if np.sum(leftMask) == 0 or np.sum(rightMask) == 0:
                    continue
                
                leftY, rightY = y[leftMask], y[rightMask]
                
                gini = (len(leftY) * giniImpurity(leftY) + 
                       len(rightY) * giniImpurity(rightY)) / len(y)
                
                if gini < bestGini:
                    bestGini = gini
                    bestSplit = (feature, value)
                    bestLeftX, bestRightX = X[leftMask], X[rightMask]
                    bestLeftY, bestRightY = leftY, rightY
        
        return bestSplit, bestLeftX, bestRightX, bestLeftY, bestRightY
    
    def buildTree(X, y, depth):
        if depth == maxDepth or len(np.unique(y)) == 1 or len(y) < 2:
            return int(np.bincount(y.astype(int)).argmax())
        
        split, leftX, rightX, leftY, rightY = bestSplit(X, y)
        
        if split is None:
            return int(np.bincount(y.astype(int)).argmax())
        
        feature, value = split
        
        leftTree = buildTree(leftX, leftY, depth + 1)
        rightTree = buildTree(rightX, rightY, depth + 1)
        
        return {'feature': feature, 'value': value, 'left': leftTree, 'right': rightTree}
    
    return buildTree(X, y, 0)

def randomForest(X, y, nEstimators=10, maxDepth=5, sampleRatio=0.8):
    """
    Random forest regressor.
    
    Parameters:
        X: features
        y: target values
        nEstimators: number of trees
        maxDepth: maximum tree depth
        sampleRatio: fraction of samples per tree
    
    Returns:
        predict function
    """
    trees = []
    
    for _ in range(nEstimators):
        idx = np.random.choice(len(X), int(len(X) * sampleRatio), replace=True)
        tree = regressionTree(X[idx], y[idx], maxDepth=maxDepth)
        trees.append(tree)
    
    def predict(XNew):
        preds = np.array([predictTree(tree, XNew) for tree in trees])
        return preds.mean(axis=0)
    
    return predict

def gradientBoosting(X, y, nEstimators=100, learningRate=0.1, maxDepth=3):
    """
    Gradient boosting regressor.
    
    Parameters:
        X: features
        y: target values
        nEstimators: number of boosting rounds
        learningRate: learning rate (shrinkage)
        maxDepth: maximum tree depth
    
    Returns:
        predict function
    """
    yPred = np.zeros_like(y, dtype=float)
    models = []
    
    for _ in range(nEstimators):
        residual = y - yPred
        tree = regressionTree(X, residual, maxDepth=maxDepth)
        models.append(tree)
        preds = predictTree(tree, X)
        yPred += learningRate * preds
    
    def predict(XNew):
        total = np.zeros(len(XNew))
        for tree in models:
            total += learningRate * predictTree(tree, XNew)
        return total
    
    return predict

def pca(X, nComponents=None):
    """
    Principal component analysis.
    
    Parameters:
        X: features (n_samples, n_features)
        nComponents: number of components to keep
    
    Returns:
        transformed data, eigenvalues, eigenvectors
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
    
    XPca = XCentered @ eigenvectors
    
    return XPca, eigenvalues, eigenvectors

def lda(X, y, nComponents=None):
    """
    Linear discriminant analysis.
    
    Parameters:
        X: features
        y: class labels
        nComponents: number of components
    
    Returns:
        transformed data, eigenvalues, eigenvectors
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
    
    if nComponents is not None:
        eigenvecs = eigenvecs[:, :nComponents]
        eigenvals = eigenvals[:nComponents]
    
    XLda = (X - meanOverall) @ eigenvecs.real
    
    return XLda, eigenvals.real, eigenvecs.real

def logisticRegression(X, y, learningRate=0.01, nIters=1000):
    """
    Logistic regression classifier.
    
    Parameters:
        X: features
        y: binary labels (0/1)
        learningRate: learning rate
        nIters: number of iterations
    
    Returns:
        weights, bias, predict function
    """
    nSamples, nFeatures = X.shape
    weights = np.zeros(nFeatures)
    bias = 0
    
    for _ in range(nIters):
        linearModel = X @ weights + bias
        yPred = 1 / (1 + np.exp(-linearModel))
        
        dw = (1 / nSamples) * (X.T @ (yPred - y))
        db = (1 / nSamples) * np.sum(yPred - y)
        
        weights -= learningRate * dw
        bias -= learningRate * db
    
    def predict(XNew):
        linearModel = XNew @ weights + bias
        yPred = 1 / (1 + np.exp(-linearModel))
        return (yPred >= 0.5).astype(int)
    
    return weights, bias, predict