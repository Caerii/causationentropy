import warnings

import numpy as np
import scipy
from scipy.special import gamma, i0, i1
from scipy.stats import nbinom
from sklearn.neighbors import KernelDensity


def l2dist(a, b):
    r"""
    Compute the Euclidean (L2) distance between two points.

    .. math::

        d(a, b) = ||a - b||_2 = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}

    Parameters
    ----------
    a, b : array-like
        Input points or vectors.

    Returns
    -------
    distance : float
        Euclidean distance between a and b.
    """
    return np.linalg.norm(a - b)


def hyperellipsoid_check(svd_Yi, Z_i):
    """
    Check if points lie within a hyperellipsoid defined by SVD components.

    This function determines whether points in Z_i fall within the unit
    hyperellipsoid defined by the singular value decomposition of Yi.

    Parameters
    ----------
    svd_Yi : tuple
        SVD decomposition (U, S, Vt) of the reference matrix.
    Z_i : array-like
        Points to test for inclusion in the hyperellipsoid.

    Returns
    -------
    inside : bool
        True if all points lie within the hyperellipsoid, False otherwise.

    Notes
    -----
    This is used in the geometric k-NN entropy estimation to assess
    the local geometric configuration of nearest neighbors.
    """
    Z_i = np.asarray(Z_i)
    if Z_i.ndim == 1:
        return bool(_hyperellipsoid_inside(svd_Yi, Z_i[None, :])[0])
    return bool(np.all(_hyperellipsoid_inside(svd_Yi, Z_i)))


def _hyperellipsoid_inside(svd_Yi, Z_i):
    r"""Test which neighbor displacement vectors lie inside a unit hyperellipsoid.

    Given the SVD :math:`Y_i = U \Sigma V^\top` of the centered neighbor cloud
    and displacement vectors :math:`Z_i` (each row points from the base sample
    to one neighbor), we map each row into ellipsoid coordinates via
    :math:`Z_i V^\top \Sigma^{-1}` and accept those whose squared norm is at
    most one. This is the batch form of the check used in Lord–Sun–Bollt's
    geometric k-NN correction.
    """
    _, S, Vt = svd_Yi
    r = len(S)
    transformed = (Z_i @ Vt.T[:, :r]) / S
    return (transformed**2).sum(axis=1) <= 1


def _singular_ratio_term(sing_Yi):
    r"""Accumulate log ratios of singular values to the leading singular value.

    When the local neighbor cloud is nearly rank-deficient, the ratio
    :math:`\sigma_\ell / \sigma_1` becomes small; we take :math:`\log` of each
    ratio (with a floor at :math:`10^{-12}`) and sum. This term appears in the
    geometric correction that adjusts standard k-NN entropy for local curvature.
    """
    if len(sing_Yi) == 0 or sing_Yi[0] <= 1e-12:
        return 0.0
    ratios = sing_Yi / sing_Yi[0]
    return float(np.sum(np.where(ratios > 1e-12, np.log(ratios), -12.0)))


def kde_entropy(X, bandwidth="silverman", kernel="gaussian"):
    r"""
    Estimate entropy using Kernel Density Estimation (KDE).

    This function computes the differential entropy of a continuous random variable
    using kernel density estimation. The entropy is defined as:

    .. math::

        H(X) = -\int f(x) \log f(x) \, dx

    where :math:`f(x)` is the probability density function estimated via KDE:

    .. math::

        \hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)

    with kernel function :math:`K` and bandwidth :math:`h`.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data for entropy estimation.
    bandwidth : str or float, default='silverman'
        Bandwidth selection method or explicit bandwidth value.
        If 'silverman', uses Silverman's rule of thumb.
    kernel : str, default='gaussian'
        Kernel function type. Options include 'gaussian', 'tophat', 'epanechnikov',
        'exponential', 'linear', 'cosine'.

    Returns
    -------
    H : float
        Estimated differential entropy in nats (natural units).

    Notes
    -----
    The KDE entropy estimator can suffer from boundary effects and may be biased
    for small sample sizes. The choice of bandwidth critically affects the estimate:

    - Too small: Undersmoothed, entropy overestimated
    - Too large: Oversmoothed, entropy underestimated

    Silverman's rule provides a reasonable default bandwidth for Gaussian-like data.
    """
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
    log_density = np.exp(kde.score_samples(X))
    Hx = -np.sum(np.log(log_density)) / len(log_density)
    return Hx


def geometric_knn_entropy(X, Xdist, k=1):
    r"""
    Estimate entropy using geometric k-nearest neighbor method.

    This function implements the geometric k-NN entropy estimator from Lord, Sun, and Bollt.
    The method estimates differential entropy by analyzing the geometric properties of
    k-nearest neighbor configurations in the data space.

    The entropy estimate is given by:

    .. math::

        H(X) = \log N + \log \frac{\pi^{d/2}}{\Gamma(1 + d/2)} + \frac{d}{N} \sum_{i=1}^{N} \log \rho_i + \text{geometric correction}

    where :math:`N` is the sample size, :math:`d` is the dimension, :math:`\rho_i` is the
    distance to the k-th nearest neighbor of point :math:`i`, and the geometric correction
    accounts for the local geometry of the nearest neighbor configuration.

    Parameters
    ----------
    X : array-like of shape (N, d)
        Input data matrix where N is the number of samples and d is the dimensionality.
    Xdist : array-like of shape (N, N)
        Pairwise distance matrix between all points in X.
    k : int, default=1
        Number of nearest neighbors to consider for entropy estimation.

    Returns
    -------
    H_X : float
        Estimated differential entropy using the geometric k-NN method.

    Notes
    -----
    This estimator is particularly effective for:

    - High-dimensional data where traditional methods may fail
    - Data with non-uniform density distributions
    - Cases where the underlying geometry is important

    The geometric correction term accounts for the local dimensionality and shape
    of the data manifold, making this estimator more robust than standard k-NN methods.

    References
    ----------
    .. [1] Lord, W.M., Sun, J., Bollt, E.M. Geometric k-nearest neighbor estimation of
           entropy and mutual information. Chaos 28, 033113 (2018).
    """
    N, d = X.shape

    # Step 1: for each sample, identify its k nearest neighbors (exclude self at index 0).
    Xknn = np.argsort(Xdist, axis=1)[:, 1 : k + 1]

    # Step 2: baseline entropy term from sample size, dimension, and k-NN radii.
    H_X = np.log(N) + np.log(np.pi ** (d / 2) / gamma(1 + d / 2))

    rows = np.arange(N)
    dists = Xdist[rows, Xknn[:, k - 1]]
    # Avoid log(0) when duplicate points sit at zero distance from their k-th neighbor.
    log_distances = np.where(dists > 1e-12, np.log(dists), -12.0)
    H_X += d / N * np.sum(log_distances)

    # Step 3: geometric correction — one local neighborhood at a time.
    # Y_i centers the base point together with its neighbors; Z_i holds the
    # displacement vectors from the base point to each neighbor.
    successful_corrections = []
    failed_count = 0
    for i in range(N):
        Y_i = X[np.append([i], Xknn[i, :]), :] - np.mean(
            X[np.append([i], Xknn[i, :]), :], axis=0
        )
        Z_i = X[Xknn[i, :], :] - X[i, :]

        try:
            svd_Yi = np.linalg.svd(Y_i)
            sing_Yi = svd_Yi[1]

            # Count how many neighbor directions fall inside the local hyperellipsoid.
            hyperellipsoid_sum = int(_hyperellipsoid_inside(svd_Yi, Z_i).sum())
            # Avoid log(0) in the hyperellipsoid term when no neighbors qualify.
            log_hyper = -np.log(max(1, hyperellipsoid_sum))
            sing_ratio_sum = _singular_ratio_term(sing_Yi)

            correction = log_hyper + sing_ratio_sum
            if np.isfinite(correction):
                successful_corrections.append(correction)
            else:
                failed_count += 1

        except (np.linalg.LinAlgError, ValueError):
            # Rank-deficient or numerically unstable neighborhoods are skipped.
            failed_count += 1

    if failed_count > 0:
        warnings.warn(
            f"Geometric correction failed for {failed_count}/{N} points. "
            f"Entropy estimate may be biased."
        )

    if successful_corrections:
        H_X += np.mean(successful_corrections)

    return H_X


_POISSON_PMF_CHUNK = 512


def _poisson_pmf_block(ks: np.ndarray, lambdas: np.ndarray) -> np.ndarray:
    """Poisson PMF for integer orders ``ks`` and rate vector ``lambdas``.

    Returns an array of shape ``(len(ks), len(lambdas))`` via one vectorized
    ``scipy.stats.poisson.pmf`` call (broadcast ``ks[:, None]`` against
    ``lambdas[None, :]``).
    """
    ks = np.asarray(ks, dtype=int)
    return scipy.stats.poisson.pmf(ks[:, None], lambdas[None, :])


def poisson_entropy(lambdas):
    r"""
    Estimate entropy for Poisson-distributed random variables.

    This function computes the entropy of Poisson random variables with given rate
    parameters. For a Poisson random variable X with parameter λ, the entropy is:

    .. math::

        H(X) = -\sum_{k=0}^{\infty} P(X = k) \log P(X = k)

    where :math:`P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}`.

    The summation is truncated when the cumulative probability reaches a specified
    tolerance to ensure numerical stability.

    Parameters
    ----------
    lambdas : array-like
        Rate parameters for the Poisson distributions. Can be scalar or array.
        Values are automatically converted to absolute values.

    Returns
    -------
    est : float or array-like
        Estimated entropy values in nats. Shape matches the input lambdas.

    Notes
    -----
    This implementation:

    - Uses adaptive truncation based on cumulative probability mass
    - Batches Poisson PMF evaluations in blocks of up to 512 orders per
      ``scipy.stats.poisson.pmf`` call
    - Handles numerical stability by setting log(0) terms to zero
    - Returns real values even if complex arithmetic is used internally

    The estimator is particularly useful for count data and discrete event processes
    where Poisson assumptions are appropriate.

    References
    ----------
    .. [1] Fish, A., Bollt, E. Interaction networks from discrete event data by Poisson
           multivariate mutual information estimation and information flow with applications
           from gene expression data. (In preparation)
    """
    lambdas = np.abs(np.asarray(lambdas, dtype=float))
    scalar_input = lambdas.ndim == 0
    lambdas = np.ravel(lambdas)

    p0 = np.exp(-lambdas)
    psum = p0.copy()
    parts = [p0.reshape(1, -1)]
    small = 1.0
    k = 1
    max_lam = float(np.max(lambdas)) if lambdas.size else 0.0

    while float(np.max(1.0 - psum)) > 1e-16 and small > 1e-75 and k <= 100_000:
        ks = np.arange(k, min(k + _POISSON_PMF_CHUNK, 100_001))
        block = _poisson_pmf_block(ks, lambdas)
        stop_at = block.shape[0]
        for i in range(block.shape[0]):
            k_curr = int(ks[i])
            psum = psum + block[i]
            if k_curr >= max_lam:
                small = min(small, float(np.min(block[i])))
            if not (float(np.max(1.0 - psum)) > 1e-16 and small > 1e-75):
                stop_at = i + 1
                break
        if stop_at == 0:
            break
        parts.append(block[:stop_at])
        if stop_at < block.shape[0]:
            break
        k = int(ks[-1]) + 1

    P = np.vstack(parts)
    with np.errstate(divide="ignore", invalid="ignore"):
        est_a = P * np.log(P)
        est_a = np.where(P > 0, est_a, 0.0)
    if P.shape[0] == 1:
        est = -np.sum(est_a)
    else:
        est = -np.sum(est_a, axis=0)
    est = np.real(est)
    if scalar_input:
        return float(np.asarray(est).reshape(()))
    return est


def poisson_joint_entropy(Cov):
    r"""
    Estimate joint entropy for multivariate Poisson distributions.

    This function computes the joint entropy of a multivariate Poisson distribution
    using the covariance matrix structure. The joint entropy decomposes into:

    .. math::

        H(\mathbf{X}) = \sum_{i} H(X_i) + \sum_{i<j} \text{Cov}(X_i, X_j)

    where the first term represents marginal entropies and the second captures
    the interaction effects through covariances.

    Parameters
    ----------
    Cov : array-like of shape (n, n)
        Covariance matrix of the multivariate Poisson distribution.
        Diagonal elements represent marginal variances (= means for Poisson).
        Off-diagonal elements represent covariances between variables.

    Returns
    -------
    joint_entropy : float
        Estimated joint entropy of the multivariate Poisson distribution.

    Notes
    -----
    This decomposition assumes a specific form for multivariate Poisson distributions
    where the interaction structure is captured through the covariance terms.

    The method:

    1. Computes marginal entropies using diagonal elements (Poisson parameters)
    2. Adds covariance contributions from off-diagonal elements

    This approach is computationally efficient for high-dimensional Poisson models.
    """
    T = np.triu(Cov, 1)
    T = np.matrix(T)
    U = np.matrix(np.diag(Cov))
    Ent1 = np.sum(poisson_entropy(U))
    Ent2 = np.sum(T)
    return Ent1 + Ent2
