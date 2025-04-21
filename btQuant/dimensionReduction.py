import numpy as np

def pca(X, n_components=None):
    """
    Performs Principal Component Analysis (PCA) on the input data.

    Parameters:
        X (array-like): The input data, where each row represents an observation and each column represents a feature.
        n_components (int, optional): The number of principal components to retain. If None, all components are returned.

    Returns:
        X_pca (array-like): The transformed data projected onto the principal components.
        eigenvalues (array): The eigenvalues corresponding to each principal component.
        eigenvectors (array): The eigenvectors (principal components) corresponding to the eigenvalues.
    """
    X_meaned = X - np.mean(X, axis=0)
    X_scaled = X_meaned / np.std(X, axis=0)
    
    covariance_matrix = np.cov(X_scaled, rowvar=False)
    
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
    
    X_pca = np.dot(X_scaled, eigenvectors)
    
    return X_pca, eigenvalues, eigenvectors

import numpy as np

def lda(X, y, n_components=None):
    """
    Performs Linear Discriminant Analysis (LDA) on the input data.

    Parameters:
        X (array-like): The input data, where each row is an observation and each column is a feature.
        y (array-like): The class labels corresponding to each observation.
        n_components (int, optional): The number of components to retain. If None, all components are returned.

    Returns:
        X_lda (array-like): The transformed data projected onto the discriminant components.
        eigenvalues (array): The eigenvalues corresponding to each component.
        eigenvectors (array): The eigenvectors (discriminant components) corresponding to the eigenvalues.
    """
    mean_overall = np.mean(X, axis=0)
    
    classes = np.unique(y)
    mean_classes = np.array([np.mean(X[y == c], axis=0) for c in classes])
    
    S_b = np.zeros((X.shape[1], X.shape[1]))
    for c, mean_class in zip(classes, mean_classes):
        n_c = X[y == c].shape[0]
        mean_diff = (mean_class - mean_overall).reshape(-1, 1)
        S_b += n_c * np.dot(mean_diff, mean_diff.T)
    
    S_w = np.zeros((X.shape[1], X.shape[1]))
    for c, mean_class in zip(classes, mean_classes):
        X_c = X[y == c]
        mean_diff = (X_c - mean_class).T
        S_w += np.dot(mean_diff, mean_diff.T)
    
    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S_w).dot(S_b))

    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    if n_components is not None:
        eigvecs = eigvecs[:, :n_components]
    
    X_lda = np.dot(X - mean_overall, eigvecs)
    
    return X_lda, eigvals, eigvecs

def tsne(X, n_components=2, perplexity=30.0, max_iter=1000, learning_rate=200):
    """
    Simplified implementation of t-SNE.

    Parameters:
        X (array-like): The input data.
        n_components (int): The number of dimensions to reduce to.
        perplexity (float): The perplexity value, controlling the balance between local and global aspects of the data.
        max_iter (int): The number of iterations for optimization.
        learning_rate (float): The learning rate for gradient descent.

    Returns:
        Y (array-like): The reduced data in the target dimension.
    """
    def compute_pairwise_affinities(X, perplexity):
        n = X.shape[0]
        pairwise_distances = np.linalg.norm(X[:, np.newaxis] - X, axis=2)
        P = np.zeros((n, n))
        
        for i in range(n):
            sigma_i = np.std(pairwise_distances[i, :])
            P[i, :] = np.exp(-pairwise_distances[i, :]**2 / (2 * sigma_i**2))
            P[i, i] = 0
        P = P / np.sum(P, axis=1, keepdims=True)
        return P
    
    Y = np.random.randn(X.shape[0], n_components)
    
    for iteration in range(max_iter):
        pairwise_distances_low = np.linalg.norm(Y[:, np.newaxis] - Y, axis=2)
        Q = 1 / (1 + pairwise_distances_low**2)
        np.fill_diagonal(Q, 0)
        Q = Q / np.sum(Q, axis=1, keepdims=True)
        
        P = compute_pairwise_affinities(X, perplexity)
        grad = 4 * np.dot((P - Q), (Y - Y[:, np.newaxis]).T) / (1 + pairwise_distances_low**2)
        
        Y -= learning_rate * grad
        
        if iteration % 100 == 0:
            print(f"Iteration {iteration}, Loss: {np.sum(P * np.log(P / Q))}")
    
    return Y

def ica(X, n_components=None):
    """
    Performs Independent Component Analysis (ICA) on the input data.

    Parameters:
        X (array-like): The input data.
        n_components (int, optional): The number of components to retain. If None, all components are returned.

    Returns:
        X_ica (array-like): The transformed data projected onto the independent components.
        mixing_matrix (array): The mixing matrix.
        unmixing_matrix (array): The unmixing matrix.
    """
    X_centered = X - np.mean(X, axis=0)
    X_whitened = X_centered / np.std(X_centered, axis=0)
    
    n_features = X_whitened.shape[1]
    W = np.random.randn(n_features, n_features)
    
    for _ in range(200):
        Y = np.dot(X_whitened, W.T)
        Y_nonlinearity = np.tanh(Y)
        grad = np.dot(X_whitened.T, Y_nonlinearity) / X_whitened.shape[0] - np.sum(1 - Y_nonlinearity**2, axis=0) * W
        W += 0.1 * grad

    if n_components is not None:
        W = W[:, :n_components]

    X_ica = np.dot(X_whitened, W.T)
    
    return X_ica, W, np.linalg.inv(W)


