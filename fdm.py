import numpy as np
from scipy.linalg import eig
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.extmath import svd_flip

import skfda
from skfda.misc.metrics import l2_distance, l1_distance, PairwiseMetric
from skfda.representation import FData


l2_functional_metric = PairwiseMetric(l2_distance)
l1_functional_metric = PairwiseMetric(l1_distance)

def rbf_kernel(X, sigma = 1.0):
    # Creation of pairwise l2 distance
    l2_functional_distances = l2_functional_metric(X)
    # Compute gaussian kernel
    # TODO: Elevar al cuadrado las distancias. => DONE.
    return np.exp(-l2_functional_distances**2/2/sigma**2)

def laplacian_kernel(X, sigma = 1.0):
    # Creation of pairwise l1 distance
    l1_functional_distances = l1_functional_metric(X)
    # Compute gaussian kernel
    # TODO: Quitar cuadrado a sigma. => DONE.
    return np.exp(-l1_functional_distances/sigma)


class FDM(TransformerMixin, BaseEstimator):
    """
        Functional Diffusion Maps.
    """
    def __init__(self,
                 n_components: (int, float) = 2 ,
                 kernel: str = 'rbf',
                 sigma: (str, float, int) = 'percentil',
                 step: int = 1,
                 alpha: float = 0.0,
                 percentil: float = 50,
    ) -> None :
        """
        Parameters
        ----------
        n_components:
            Embedded space dimension.
        kernel:
            kernel type to compute the affinity matrix.
        sigma:
            Sigma parameter of rbf kernel.
        step:
            Number of steps to advance in the Markov chain.
        alpha:
            Normalization parameter to control the density influence.
        percentil:
            Percentile vale to calculate sigma used when self.sigma == 'percentil'.

        """

        self.n_components = n_components
        self.kernel = kernel
        self.sigma = sigma
        self.step = step
        self.alpha = alpha
        self.percentil = percentil

    def choose_n_components(self, eigenvals):
        """ Initialization of the embedding dimension."""

        if isinstance(self.n_components, int):
            self.n_components_ = self.n_components
        elif isinstance(self.n_components, float):
            if self.step == 0:
                self.n_components_ = 1
            else:
                self.n_components_ = np.count_nonzero(np.abs(eigenvals)**self.step >self.n_components*np.abs(eigenvals[0])**self.step) - 1
        else:
            raise ValueError(("%s is not a valid 'n_components' parameter. "
                              "Expected int of float.") % self.n_components)

    def compute_sigma(self, X):
        """ Compute sigma value for the affinity matrix distinguishing each possible method. """
        if self.sigma == 'median':
            self.sigma = np.percentile(l2_functional_metric(X), 50)
        elif self.sigma == 'maximum':
            self.sigma = np.percentile(l2_functional_metric(X), 100)
        elif self.sigma == 'percentil':
            self.sigma_ = np.percentile(l2_functional_metric(X), self.percentil)
        elif (isinstance(self.sigma, float) or isinstance(self.sigma, int)) and self.sigma > 0.0:
            self.sigma_ = self.sigma
        else:
            raise ValueError(("%s is not a valid sigma method. "
                              "Expected 'median', 'maximum', 'percentile' or a direct positive sigma value.") % self.sigma)


    def compute_affinity_matrix(self,
                                X: FData,
    ) -> np.ndarray:
        """
        Compute affinity matrix for X data.
        Parameters
        ----------
        X:
            The functional data object to be analysed
        Returns
        -------
        Kernel matrix:

        """
        if not hasattr(self, "sigma_"):
            self.compute_sigma(X)

        if self.kernel == 'rbf':
            return rbf_kernel(X, sigma = self.sigma_)
        elif self.kernel == 'laplacian':
            return laplacian_kernel(X, sigma = self.sigma_)
        else:
            raise ValueError(("%s is not a valid kernel function. "
                              "Expected 'rbf' or 'laplacian' function."))


    def fit(self,
            X: FData,
            y: FData = None
    ):
        """
        Compute the embedding vectors for data X
        Parameters
        ----------
        X:
            The functional data object to be analysed
        y :
            Ignored
        Returns
        -------
        self :
            Returns an instance of self.
        """

        # Kernel definition with affinity matrix
        K = self.compute_affinity_matrix(X)

        # Compute distance degree matrix with density
        d = np.sum(K, axis=1)**self.alpha

        # Compute density normalization
        K_alpha = K / np.outer(d, d)

        # Compute distance matrix with K normalized
        d_alpha = np.sum(K_alpha, axis=1)

        # Compute transition probability matrix
        P = K_alpha / d_alpha [: , np.newaxis ]

        # Compute spectral descomposition
        w, V = eig(P)
        idx = np.argsort(w)[::-1]
        eigenvals = np.real(w[idx])
        eigenvecs = V[:, idx]

        # Choose n components
        self.choose_n_components(eigenvals[1:])
        self.eigenvecs = eigenvecs[:, 1:self.n_components_ + 1]
        self.eigenvals = eigenvals[1:self.n_components_ + 1]

        # Deterministic sign.
        self.eigenvecs, _ = svd_flip(self.eigenvecs, np.zeros_like(self.eigenvecs).T)

        return self

    def fit_transform(self,
                      X: FData,
                      y: FData = None
    ) -> np.ndarray:
        """
        Compute the embedding vectors for data X and transform X.
        Parameters
        ----------
        X:
            The functional data object to be analysed
        y :
            Ignored
        Returns
        -------
        Embedding :
            Array-like, shape (n_samples, n_components)
        """

        # fit training set
        self.fit(X)

        # Compute the embedding
        self.embedding = (self.eigenvals**self.step)*self.eigenvecs

        return self.embedding
