# functional-diffusion-maps

This code defines a Python implementation of the Functional Diffusion Maps (FDM) algorithm. The FDM is a manifold learning technique designed for functional data. It is based on the construction of a Markov chain on a graph of pairwise distances between functional data, where the transition probability is determined by the kernel function. The eigenvectors of the graph Laplacian matrix are then used to embed the functional data into a lower-dimensional space.

The code imports necessary modules and functions, including NumPy, SciPy, scikit-learn (for the base estimator and transformer), and skfda (for functional data analysis). The kernel functions used for computing the affinity matrix include the radial basis function (RBF) kernel and the Laplacian kernel. The former uses the L2 distance metric to compute pairwise distances, while the latter uses the L1 distance metric.
