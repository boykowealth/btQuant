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
