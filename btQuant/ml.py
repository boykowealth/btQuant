import numpy as np

def regression_tree(X, y, max_depth=5):
    """
    A simple Regression Tree implementation.
    
    Parameters:
        X (array-like): The input data (features).
        y (array-like): The target values.
        max_depth (int): The maximum depth of the tree.
    
    Returns:
        dict: The regression tree.
    """
    def mean_squared_error(y):
        return np.mean((y - np.mean(y))**2)
    
    def build_tree(X, y, depth):
        if depth == max_depth or len(np.unique(y)) == 1:
            return np.mean(y)
        
        best_mse = float('inf')
        best_split = None
        best_left_y = None
        best_right_y = None
        best_left_X = None
        best_right_X = None
        best_feature = None
        best_value = None
        
        n_features = X.shape[1]
        for feature in range(n_features):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask
                left_X = X[left_mask]
                right_X = X[right_mask]
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                mse = (mean_squared_error(left_y) * len(left_y) + 
                       mean_squared_error(right_y) * len(right_y)) / len(y)
                
                if mse < best_mse:
                    best_mse = mse
                    best_split = (feature, value)
                    best_left_X = left_X
                    best_right_X = right_X
                    best_left_y = left_y
                    best_right_y = right_y
        
        feature, value = best_split
        left_tree = build_tree(best_left_X, best_left_y, depth + 1)
        right_tree = build_tree(best_right_X, best_right_y, depth + 1)
        
        return {'feature': feature, 'value': value, 'left': left_tree, 'right': right_tree}
    
    return build_tree(X, y, 0)

def isolation_forest(X, n_trees=100, max_samples=None, max_depth=10):
    """
    A simple Isolation Forest implementation.
    
    Parameters:
        X (array-like): The input data (features).
        n_trees (int): Number of isolation trees to build.
        max_samples (int): Number of samples to use for each tree (optional).
        max_depth (int): Maximum depth of the tree.
    
    Returns:
        list: The isolation trees.
    """
    def build_isolation_tree(X, depth):
        if depth >= max_depth or len(np.unique(X, axis=0)) == 1:
            return np.mean(X, axis=0)
        
        feature = np.random.randint(0, X.shape[1])
        min_value, max_value = X[:, feature].min(), X[:, feature].max()
        split_value = np.random.uniform(min_value, max_value)
        
        left_mask = X[:, feature] <= split_value
        right_mask = ~left_mask
        
        left_tree = build_isolation_tree(X[left_mask], depth + 1)
        right_tree = build_isolation_tree(X[right_mask], depth + 1)
        
        return {'feature': feature, 'value': split_value, 'left': left_tree, 'right': right_tree}

    trees = []
    n_samples = X.shape[0]
    if max_samples is None:
        max_samples = n_samples
    for _ in range(n_trees):
        sampled_indices = np.random.choice(n_samples, max_samples, replace=False)
        sampled_X = X[sampled_indices]
        tree = build_isolation_tree(sampled_X, 0)
        trees.append(tree)
    
    return trees

def kmeans(X, k=3, max_iters=100):
    """
    A simple K-Means Clustering implementation.
    
    Parameters:
        X (array-like): The input data (features).
        k (int): The number of clusters.
        max_iters (int): Maximum number of iterations.
    
    Returns:
        tuple: (centroids, labels) - centroids of clusters and the labels assigned to each data point.
    """
    n_samples, n_features = X.shape
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    
    for _ in range(max_iters):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(k)])
        
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return centroids, labels


def knn(X_train, y_train, X_test, k=3):
    """
    A simple K-Nearest Neighbors (KNN) implementation.
    
    Parameters:
        X_train (array-like): The training data (features).
        y_train (array-like): The training labels.
        X_test (array-like): The test data (features).
        k (int): The number of nearest neighbors to consider.
    
    Returns:
        array: Predicted labels for the test data.
    """
    def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    predictions = []
    
    for test_point in X_test:
        distances = [euclidean_distance(test_point, train_point) for train_point in X_train]
        sorted_indices = np.argsort(distances)
        nearest_neighbors = y_train[sorted_indices[:k]]
        most_common = np.bincount(nearest_neighbors).argmax()
        predictions.append(most_common)
    
    return np.array(predictions)

def gaussian_naive_bayes(X_train, y_train, X_test):
    """
    A simple Gaussian Naive Bayes classifier.
    
    Parameters:
        X_train (array-like): The training data (features).
        y_train (array-like): The training labels.
        X_test (array-like): The test data (features).
    
    Returns:
        array: Predicted labels for the test data.
    """
    def gaussian_pdf(x, mean, std):
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
    
    classes = np.unique(y_train)
    mean_std = {cls: {'mean': np.mean(X_train[y_train == cls], axis=0), 
                      'std': np.std(X_train[y_train == cls], axis=0)} for cls in classes}
    
    predictions = []
    
    for test_point in X_test:
        class_probabilities = {}
        
        for cls in classes:
            likelihood = np.prod(gaussian_pdf(test_point, mean_std[cls]['mean'], mean_std[cls]['std']))
            class_probabilities[cls] = likelihood
        
        predicted_class = max(class_probabilities, key=class_probabilities.get)
        predictions.append(predicted_class)
    
    return np.array(predictions)

def decision_tree(X, y, max_depth=5):
    """
    A simple Decision Tree classifier.
    
    Parameters:
        X (array-like): The input data (features).
        y (array-like): The target labels.
        max_depth (int): The maximum depth of the tree.
    
    Returns:
        dict: The decision tree.
    """
    def gini_impurity(y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def best_split(X, y):
        best_gini = float('inf')
        best_split = None
        best_left_y, best_right_y = None, None
        best_left_X, best_right_X = None, None
        
        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_mask = X[:, feature] <= value
                right_mask = ~left_mask
                left_y, right_y = y[left_mask], y[right_mask]
                
                gini = (len(left_y) * gini_impurity(left_y) + len(right_y) * gini_impurity(right_y)) / len(y)
                
                if gini < best_gini:
                    best_gini = gini
                    best_split = (feature, value)
                    best_left_X, best_right_X = X[left_mask], X[right_mask]
                    best_left_y, best_right_y = left_y, right_y
        
        return best_split, best_left_X, best_right_X, best_left_y, best_right_y
    
    def build_tree(X, y, depth):
        if depth == max_depth or len(np.unique(y)) == 1:
            return np.bincount(y).argmax()
        
        split, left_X, right_X, left_y, right_y = best_split(X, y)
        feature, value = split
        
        left_tree = build_tree(left_X, left_y, depth + 1)
        right_tree = build_tree(right_X, right_y, depth + 1)
        
        return {'feature': feature, 'value': value, 'left': left_tree, 'right': right_tree}
    
    return build_tree(X, y, 0)

def random_forest(X, y, n_estimators=10, max_depth=5, sample_ratio=0.8):
    def regression_tree(X, y, depth):
        if depth == 0 or len(set(y)) == 1:
            return np.mean(y)
        best_feature, best_value, best_loss = None, None, float('inf')
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for val in values:
                left = y[X[:, feature] <= val]
                right = y[X[:, feature] > val]
                if len(left) == 0 or len(right) == 0:
                    continue
                loss = np.var(left)*len(left) + np.var(right)*len(right)
                if loss < best_loss:
                    best_feature, best_value, best_loss = feature, val, loss
        if best_feature is None:
            return np.mean(y)
        left_tree = regression_tree(X[X[:, best_feature] <= best_value],
                                    y[X[:, best_feature] <= best_value], depth - 1)
        right_tree = regression_tree(X[X[:, best_feature] > best_value],
                                     y[X[:, best_feature] > best_value], depth - 1)
        return {'feature': best_feature, 'value': best_value, 'left': left_tree, 'right': right_tree}

    def predict_tree(tree, x):
        while isinstance(tree, dict):
            if x[tree['feature']] <= tree['value']:
                tree = tree['left']
            else:
                tree = tree['right']
        return tree

    trees = []
    for _ in range(n_estimators):
        idx = np.random.choice(len(X), int(len(X) * sample_ratio), replace=True)
        X_sample, y_sample = X[idx], y[idx]
        tree = regression_tree(X_sample, y_sample, max_depth)
        trees.append(tree)

    def predict(X_new):
        preds = np.array([[predict_tree(tree, x) for tree in trees] for x in X_new])
        return preds.mean(axis=1)

    return predict

def gradient_boosting(X, y, n_estimators=100, learning_rate=0.1, max_depth=3):
    def regression_tree(X, y, depth):
        if depth == 0 or len(set(y)) == 1:
            return np.mean(y)
        best_feature, best_value, best_loss = None, None, float('inf')
        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            for val in values:
                left = y[X[:, feature] <= val]
                right = y[X[:, feature] > val]
                if len(left) == 0 or len(right) == 0:
                    continue
                loss = np.var(left)*len(left) + np.var(right)*len(right)
                if loss < best_loss:
                    best_feature, best_value, best_loss = feature, val, loss
        if best_feature is None:
            return np.mean(y)
        left_tree = regression_tree(X[X[:, best_feature] <= best_value],
                                    y[X[:, best_feature] <= best_value], depth - 1)
        right_tree = regression_tree(X[X[:, best_feature] > best_value],
                                     y[X[:, best_feature] > best_value], depth - 1)
        return {'feature': best_feature, 'value': best_value, 'left': left_tree, 'right': right_tree}

    def predict_tree(tree, x):
        while isinstance(tree, dict):
            if x[tree['feature']] <= tree['value']:
                tree = tree['left']
            else:
                tree = tree['right']
        return tree

    y_pred = np.zeros_like(y)
    models = []

    for _ in range(n_estimators):
        residual = y - y_pred
        tree = regression_tree(X, residual, max_depth)
        models.append(tree)
        preds = np.array([predict_tree(tree, x) for x in X])
        y_pred += learning_rate * preds

    def predict(X_new):
        total = np.zeros(len(X_new))
        for tree in models:
            total += learning_rate * np.array([predict_tree(tree, x) for x in X_new])
        return total

    return predict


