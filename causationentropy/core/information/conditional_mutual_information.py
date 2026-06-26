import numpy as np
from scipy.special import digamma

from causationentropy.core.information.distance_cache import (
    cached_cdist,
    cached_corrcoef,
    correlation_log_det_subset,
    supports_kd_tree,
    tree_knn_distances,
    tree_neighbors_within_distance,
)
from causationentropy.core.information.entropy import (
    geometric_knn_entropy,
    kde_entropy,
    poisson_entropy,
    poisson_joint_entropy,
)
from causationentropy.core.information.mutual_information import (
    gaussian_mutual_information,
    geometric_knn_mutual_information,
    kde_mutual_information,
    knn_mutual_information,
)


def gaussian_conditional_mutual_information(X, Y, Z=None):
    r"""
    Compute conditional mutual information for multivariate Gaussian variables.

    For multivariate Gaussian variables, the conditional mutual information has
    a closed-form expression using covariance matrix determinants:

    .. math::

        I(X; Y | Z) = \frac{1}{2} \log \frac{|\Sigma_{XZ}| |\Sigma_{YZ}|}{|\Sigma_Z| |\Sigma_{XYZ}|}

    This can also be expressed as:

    .. math::

        I(X; Y | Z) = \frac{1}{2} [\log |\Sigma_{XZ}| + \log |\Sigma_{YZ}| - \log |\Sigma_Z| - \log |\Sigma_{XYZ}|]

    where :math:`\Sigma_{\cdot}` denotes the covariance matrix of the subscripted variables.

    Parameters
    ----------
    X : array-like of shape (N, k_x)
        First variable with N samples and k_x features.
    Y : array-like of shape (N, k_y)
        Second variable with N samples and k_y features.
    Z : array-like of shape (N, k_z) or None
        Conditioning variable with N samples and k_z features.
        If None, computes marginal mutual information I(X;Y).

    Returns
    -------
    I : float
        Conditional mutual information in nats.

    Notes
    -----
    This implementation uses log-determinants of correlation matrices for
    numerical stability, employing the signed log-determinant function
    to handle potential numerical issues.

    The Gaussian assumption implies that:
    - All conditional dependencies are captured by linear relationships
    - Higher-order moments beyond covariance carry no information
    - The estimator is exact under Gaussianity

    For non-Gaussian data, this estimator provides a lower bound on the
    true conditional mutual information.

    Implementation note
    -------------------
    Stack ``[X | Y | Z]`` once, cache ``corrcoef(W)``, and extract the four
    log-determinants required by the formula from column-index blocks. The
    index layout is fixed by concatenation order so every term refers to the
    same underlying correlation matrix.
    """
    if Z is None:
        return gaussian_mutual_information(X, Y)

    kx, ky, kz = X.shape[1], Y.shape[1], Z.shape[1]
    W = np.hstack((X, Y, Z))
    C = cached_corrcoef(W)
    # Partition columns: X | Y | Z in concatenation order.
    ix = np.arange(kx)
    iy = np.arange(kx, kx + ky)
    iz = np.arange(kx + ky, kx + ky + kz)
    ixz = np.concatenate((ix, iz))
    iyz = np.concatenate((iy, iz))
    ixyz = np.concatenate((ix, iy, iz))

    # I(X;Y|Z) = 1/2 [ log|XZ| + log|YZ| - log|Z| - log|XYZ| ] in correlation form.
    cmi = 0.5 * (
        correlation_log_det_subset(C, ixz)
        + correlation_log_det_subset(C, iyz)
        - correlation_log_det_subset(C, iz)
        - correlation_log_det_subset(C, ixyz)
    )
    return cmi


def gaussian_conditional_mutual_information_batch(X_candidates, Y, Z=None):
    r"""Evaluate Gaussian CMI for many candidate predictors in one correlation pass.

    Forward selection repeatedly asks the same question with a fixed target ``Y``
    and conditioning set ``Z``, varying only which single column of ``X`` is the
    candidate predictor. Each question is

    .. math::

        I(X_j;\, Y \mid Z),

    and naively we would build a separate correlation matrix for every ``j``. The
    candidates share ``Y`` and ``Z``, however, so we stack all candidate columns
    into ``X_candidates``, form one matrix :math:`W = [X_{\text{cand}} \mid Y \mid Z]`,
    call :func:`cached_corrcoef` once, and read each answer from a principal minor
    of the same :math:`C`.

    Parameters
    ----------
    X_candidates : array-like of shape (N, n_cand)
        Each column is one candidate predictor (typically all remaining variables
        in a forward-selection step).
    Y : array-like of shape (N, k_y)
        Target block held fixed across candidates.
    Z : array-like of shape (N, k_z) or None
        Conditioning block held fixed across candidates. ``None`` or empty yields
        marginal mutual information :math:`I(X_j; Y)` for each column.

    Returns
    -------
    cmi : ndarray of shape (n_cand,)
        Conditional (or marginal) mutual information in nats, one value per column
        of ``X_candidates``.

    Notes
    -----
    Numerically identical to calling :func:`gaussian_conditional_mutual_information`
    on each one-column slice; this entry point exists to amortize ``corrcoef`` cost
    during discovery when ``information='gaussian'``.
    """
    X_candidates = np.asarray(X_candidates)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_cand = X_candidates.shape[1]
    if n_cand == 0:
        return np.array([])

    ky = Y.shape[1]
    Z_empty = Z is None or np.asarray(Z).size == 0

    if Z_empty:
        # Marginal MI I(X_j; Y) for each candidate column j.
        W = np.hstack((X_candidates, Y))
        C = cached_corrcoef(W)
        iy = np.arange(n_cand, n_cand + ky)
        results = np.empty(n_cand)
        for j in range(n_cand):
            ix = np.array([j])
            ixy = np.concatenate((ix, iy))
            results[j] = 0.5 * (
                correlation_log_det_subset(C, ix)
                + correlation_log_det_subset(C, iy)
                - correlation_log_det_subset(C, ixy)
            )
        return results

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    kz = Z.shape[1]

    W = np.hstack((X_candidates, Y, Z))
    C = cached_corrcoef(W)
    iy = np.arange(n_cand, n_cand + ky)
    iz = np.arange(n_cand + ky, n_cand + ky + kz)

    results = np.empty(n_cand)
    for j in range(n_cand):
        ix = np.array([j])
        ixz = np.concatenate((ix, iz))
        iyz = np.concatenate((iy, iz))
        ixyz = np.concatenate((ix, iy, iz))
        results[j] = 0.5 * (
            correlation_log_det_subset(C, ixz)
            + correlation_log_det_subset(C, iyz)
            - correlation_log_det_subset(C, iz)
            - correlation_log_det_subset(C, ixyz)
        )
    return results


def kde_conditional_mutual_information(
    X, Y, Z, bandwidth="silverman", kernel="gaussian"
):
    """
    Estimate conditional mutual information using Kernel Density Estimation.

    This function computes conditional mutual information using the entropy decomposition:

    .. math::

        I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    where each entropy term is estimated using kernel density estimation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, reduces to marginal mutual information.
    bandwidth : str or float, default='silverman'
        Bandwidth parameter for KDE.
    kernel : str, default='gaussian'
        Kernel function for density estimation.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.

    Notes
    -----
    The KDE approach can capture nonlinear conditional dependencies but suffers from:
    - Curse of dimensionality for high-dimensional conditioning sets
    - Bandwidth selection sensitivity
    - Computational complexity scaling with sample size

    Consider k-NN methods for high-dimensional problems or large datasets.
    """
    if Z is None:
        I = kde_mutual_information(X, Y, bandwidth=bandwidth, kernel=kernel)
    else:
        XZ = np.hstack((X, Z))
        YZ = np.hstack((Y, Z))
        XYZ = np.hstack((X, Y, Z))

        # Compute the entropies
        Hz = kde_entropy(Z, bandwidth=bandwidth, kernel=kernel)
        Hxz = kde_entropy(XZ, bandwidth=bandwidth, kernel=kernel)
        Hyz = kde_entropy(YZ, bandwidth=bandwidth, kernel=kernel)
        Hxyz = kde_entropy(XYZ, bandwidth=bandwidth, kernel=kernel)
        I = Hxz + Hyz - Hxyz - Hz

    return I


def knn_conditional_mutual_information(X, Y, Z, metric="minkowski", k=1, kd_tree=True):
    """
    Estimate conditional mutual information using k-nearest neighbor method.

    This function implements conditional mutual information estimation using
    the relationship:

    .. math::

        I(X; Y | Z) = I(X, Y) - I(X, Y; Z)

    where both mutual information terms are estimated using the KSG k-NN estimator.

    The approach leverages the fact that:

    .. math::

        I(X; Y | Z) = I(X; Y) - I(X; Y | Z)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, computes marginal mutual information.
    metric : str, default='minkowski'
        Distance metric for k-NN calculations.
    k : int, default=1
        Number of nearest neighbors.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.

    Notes
    -----
    This implementation uses the decomposition approach rather than direct
    conditional MI estimation. The accuracy depends on:

    - Quality of marginal MI estimates
    - Dimensionality of the joint space
    - Sample size relative to effective dimensionality

    References
    ----------
    .. [1] Kraskov, A., Stögbauer, H., Grassberger, P. Estimating mutual information.
           Physical Review E 69, 066138 (2004).
    """
    if Z is None:
        return knn_mutual_information(X, Y, metric=metric, k=k, kd_tree=kd_tree)
    else:
        JS = np.column_stack((X, Y, Z))
        if kd_tree and supports_kd_tree(metric) and metric != "minkowski":
            distances_js, _ = tree_knn_distances(JS, k=k, metric=metric)
            epsilon = distances_js[:, k - 1]
            nxz = tree_neighbors_within_distance(
                np.column_stack((X, Z)), epsilon, metric=metric
            )
            nyz = tree_neighbors_within_distance(
                np.column_stack((Y, Z)), epsilon, metric=metric
            )
            nz = tree_neighbors_within_distance(Z, epsilon, metric=metric)
        else:
            if metric == "minkowski":
                D = np.sort(cached_cdist(JS, metric=metric, p=k + 1), axis=1)[:, k]
            else:
                D = np.sort(cached_cdist(JS, metric=metric), axis=1)[:, k]
            epsilon = D
            Dxz = cached_cdist(np.column_stack((X, Z)), metric=metric)
            nxz = np.sum(Dxz < epsilon[:, None], axis=1) - 1
            Dyz = cached_cdist(np.column_stack((Y, Z)), metric=metric)
            nyz = np.sum(Dyz < epsilon[:, None], axis=1) - 1
            Dz = cached_cdist(Z, metric=metric)
            nz = np.sum(Dz < epsilon[:, None], axis=1) - 1

        I = digamma(k) - np.mean(digamma(nxz + 1) + digamma(nyz + 1) - digamma(nz + 1))
        return I


def geometric_knn_conditional_mutual_information(
    X, Y, Z, metric="euclidean", k=1, kd_tree=True
):
    """
    Estimate conditional mutual information using geometric k-nearest neighbor method.

    This function applies the geometric k-NN entropy estimator to compute
    conditional mutual information via the entropy decomposition:

    .. math::

        I(X; Y | Z) = H_{\text{geom}}(X, Z) + H_{\text{geom}}(Y, Z) - H_{\text{geom}}(Z) - H_{\text{geom}}(X, Y, Z)

    The geometric correction accounts for local manifold structure, providing
    improved estimates for data with non-uniform density or intrinsic dimensionality
    lower than the ambient space.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, computes marginal mutual information.
    metric : str, default='euclidean'
        Distance metric for neighbor calculations.
    k : int, default=1
        Number of nearest neighbors.

    Returns
    -------
    I : float
        Estimated conditional mutual information using geometric k-NN method.

    Notes
    -----
    The geometric approach is particularly effective for:
    - Data on lower-dimensional manifolds
    - Non-uniform density distributions
    - Cases where local geometric structure is important

    The method accounts for the effective local dimensionality through
    geometric corrections to the standard k-NN entropy estimates.

    References
    ----------
    .. [1] Lord, W.M., Sun, J., Bollt, E.M. Geometric k-nearest neighbor estimation of
           entropy and mutual information. Chaos 28, 033113 (2018).
    """

    if Z is None:
        return geometric_knn_mutual_information(
            X, Y, metric=metric, k=k, kd_tree=kd_tree
        )
    YZdist = cached_cdist(np.hstack((Y, Z)), metric=metric)
    XZdist = cached_cdist(np.hstack((X, Z)), metric=metric)
    XYZdist = cached_cdist(np.hstack((X, Y, Z)), metric=metric)
    Zdist = cached_cdist(Z, metric=metric)
    HZ = geometric_knn_entropy(Z, Zdist, k)
    HXZ = geometric_knn_entropy(np.hstack((X, Z)), XZdist, k)
    HYZ = geometric_knn_entropy(np.hstack((Y, Z)), YZdist, k)
    HXYZ = geometric_knn_entropy(np.hstack((X, Y, Z)), XYZdist, k)
    cmi = HXZ + HYZ - HXYZ - HZ
    return cmi


def _poisson_marginal_mi_from_correlation(SXY: np.ndarray, ix: np.ndarray, iy: np.ndarray) -> float:
    """Poisson MI from a joint correlation matrix and column index blocks."""
    idx = np.concatenate((ix, iy))
    block = SXY[np.ix_(idx, idx)]
    l_est = block - np.diag(np.diag(block))
    np.fill_diagonal(block, np.diagonal(block) - np.sum(l_est, axis=0))
    dcov = np.diag(block) + np.sum(l_est, axis=0)
    tf = poisson_joint_entropy(block)
    ft = np.sum(poisson_entropy(dcov))
    return float(ft - tf)


def _poisson_conditional_cmi_from_correlation(
    SXYZ: np.ndarray, ix: np.ndarray, iy: np.ndarray, iz: np.ndarray
) -> float:
    """Poisson CMI from one joint correlation matrix and X/Y/Z column indices.

    The Fish–Sun–Bollt Poisson estimator rearranges off-diagonal covariance
    mass in the correlation matrix before evaluating joint entropies. This
    helper performs that rearrangement using arbitrary column indices so we can
    evaluate many single-column candidates against the same cached ``SXYZ``.
    """
    ix = np.asarray(ix, dtype=int)
    iy = np.asarray(iy, dtype=int)
    iz = np.asarray(iz, dtype=int)

    SS = SXYZ.copy()
    Sa = SXYZ - np.diag(np.diag(SXYZ))
    np.fill_diagonal(SS, np.diagonal(SS) - np.diag(Sa))
    # Move cross-block correlation mass onto X/X and Y/Y blocks (Fish–Sun–Bollt step).
    SS[np.ix_(ix, ix)] = SS[np.ix_(ix, ix)] + SXYZ[np.ix_(ix, iy)]
    SS[np.ix_(iy, iy)] = SS[np.ix_(iy, iy)] + SXYZ[np.ix_(iy, ix)]

    yz_idx = np.concatenate((iy, iz))
    xz_idx = np.concatenate((ix, iz))
    s_est_yz = SS[np.ix_(yz_idx, yz_idx)]
    s_est_xz = SS[np.ix_(xz_idx, xz_idx)]
    hyz = poisson_joint_entropy(s_est_yz)
    hz = poisson_joint_entropy(SS[np.ix_(iz, iz)])
    # H(X,Y,Z) uses the original correlation matrix with diagonal mass removed (Sa).
    hxyz = poisson_joint_entropy(SXYZ - np.diag(Sa))
    hxz = poisson_joint_entropy(s_est_xz)
    h_yz = hyz - hz
    h_xyz = hxyz - hxz
    return float(h_xyz - h_yz)


def poisson_conditional_mutual_information_batch(X_candidates, Y, Z=None):
    r"""Evaluate Poisson CMI for many candidate predictors from one ``corrcoef`` pass.

    Forward selection with ``information='poisson'`` asks the same conditional
    information question as the Gaussian batch path, but the Fish–Sun–Bollt
    estimator works from a joint correlation matrix and block-wise entropy
    decompositions. We stack all candidates with ``Y`` and ``Z``, cache one
    ``corrcoef`` on the full block, then extract the principal submatrix for
    each candidate ``[X_j, Y, Z]`` before applying the Poisson rearrangement.
    """
    X_candidates = np.asarray(X_candidates)
    Y = np.asarray(Y)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    n_cand = X_candidates.shape[1]
    if n_cand == 0:
        return np.array([])

    ky = Y.shape[1]
    Z_empty = Z is None or np.asarray(Z).size == 0

    if Z_empty:
        W = np.hstack((X_candidates, Y))
        SXY = cached_corrcoef(W)
        iy_local = np.arange(1, 1 + ky)
        results = np.empty(n_cand)
        for j in range(n_cand):
            # Column j is candidate X_j; Y columns follow all candidates in W.
            cols = np.concatenate(([j], np.arange(n_cand, n_cand + ky)))
            sub = SXY[np.ix_(cols, cols)]
            # Indices 0 and 1..ky refer to rows within sub, not W.
            results[j] = _poisson_marginal_mi_from_correlation(
                sub, np.array([0]), iy_local
            )
        return results

    Z = np.asarray(Z)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    kz = Z.shape[1]

    W = np.hstack((X_candidates, Y, Z))
    SXYZ = cached_corrcoef(W)
    iy_local = np.arange(1, 1 + ky)
    iz_local = np.arange(1 + ky, 1 + ky + kz)

    results = np.empty(n_cand)
    for j in range(n_cand):
        # Principal submatrix for [X_j, Y, Z] — joint entropy must not include
        # other candidates' cross-correlations (unlike a naive index into SXYZ).
        cols = np.concatenate(
            ([j], np.arange(n_cand, n_cand + ky), np.arange(n_cand + ky, n_cand + ky + kz))
        )
        sub = SXYZ[np.ix_(cols, cols)]
        results[j] = _poisson_conditional_cmi_from_correlation(
            sub, np.array([0]), iy_local, iz_local
        )
    return results


def poisson_conditional_mutual_information(X, Y, Z):
    """
    Estimate conditional mutual information for multivariate Poisson distributions.

    This function computes conditional mutual information for discrete count data
    assuming Poisson distributions. The estimation uses the covariance structure
    of the multivariate Poisson distribution:

    .. math::

        I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    where entropies are computed using Poisson-specific formulations that account
    for the discrete nature and parameter structure of Poisson variables.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        Count data from first Poisson variables.
    Y : array-like of shape (n_samples, n_features_y)
        Count data from second Poisson variables.
    Z : array-like of shape (n_samples, n_features_z) or None
        Count data from conditioning Poisson variables.
        If None, computes marginal mutual information.

    Returns
    -------
    I : float
        Estimated conditional mutual information for Poisson data.

    Notes
    -----
    This implementation is specifically designed for discrete count data where:
    - Variables follow Poisson distributions
    - Dependencies are captured through covariance structure
    - Joint distributions maintain Poisson-like properties

    Applications include:
    - Gene expression count data
    - Event occurrence data
    - Discrete interaction networks
    - Epidemiological count models

    Implementation note
    -------------------
    The joint correlation matrix ``SXYZ`` is obtained via
    :func:`cached_corrcoef` so repeated Poisson CMI evaluations inside
    discovery reuse the same ``corrcoef`` result when the data block is unchanged.

    References
    ----------
    .. [1] Fish, A., Sun, J., Bollt, E. Interaction networks from discrete event data by
           Poisson multivariate mutual information estimation and information flow with
           applications from gene expression data. (In preparation)
    """

    if Z is None:
        SXY = cached_corrcoef(np.hstack((X, Y)))
        ix = np.arange(X.shape[1])
        iy = np.arange(X.shape[1], X.shape[1] + Y.shape[1])
        return _poisson_marginal_mi_from_correlation(SXY, ix, iy)

    XYZ = np.concatenate((X, Y, Z), axis=1)
    SXYZ = cached_corrcoef(XYZ)
    kx, ky, kz = X.shape[1], Y.shape[1], Z.shape[1]
    ix = np.arange(kx)
    iy = np.arange(kx, kx + ky)
    iz = np.arange(kx + ky, kx + ky + kz)
    return _poisson_conditional_cmi_from_correlation(SXYZ, ix, iy, iz)


def conditional_mutual_information(
    X,
    Y,
    Z=None,
    method="gaussian",
    metric="euclidean",
    k=6,
    bandwidth="silverman",
    kernel="gaussian",
    kd_tree=True,
):
    """
    Compute conditional mutual information using specified estimation method.

    This function provides a unified interface for computing conditional mutual information
    I(X;Y|Z) using various estimation approaches. The choice of method depends on the
    data type, dimensionality, and distributional assumptions.

    Conditional mutual information quantifies the information shared between X and Y
    when conditioning on Z:

    .. math::

        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)

    Equivalently:

    .. math::

        I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, computes marginal mutual information I(X;Y).
    method : str, default='gaussian'
        Estimation method. Available options:

        - 'gaussian': Assumes multivariate Gaussian distributions
        - 'kde' or 'kernel_density': Kernel density estimation
        - 'knn': k-nearest neighbor (KSG) estimator
        - 'geometric_knn': Geometric k-NN with manifold corrections
        - 'poisson': For discrete count data with Poisson assumptions

    metric : str, default='euclidean'
        Distance metric for k-NN based methods.
    k : int, default=1
        Number of nearest neighbors for k-NN methods.
    bandwidth : str or float, default='silverman'
        Bandwidth parameter for KDE methods.
    kernel : str, default='gaussian'
        Kernel function for KDE methods.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.

    Raises
    ------
    ValueError
        If an unsupported method is specified.

    Notes
    -----
    **Method Selection Guidelines:**

    - **Gaussian**: Best for linear relationships, exact under Gaussianity
    - **KDE**: Good for smooth nonlinear dependencies, curse of dimensionality
    - **k-NN**: Robust for moderate dimensions, adapts to local density
    - **Geometric k-NN**: Effective for manifold data with intrinsic structure
    - **Poisson**: Specifically for discrete count data

    **Computational Complexity:**
    - Gaussian: O(n³) for matrix operations
    - KDE: O(n²) for density evaluation
    - k-NN: O(n² log n) for neighbor finding

    **Sample Size Requirements:**
    - Increase with dimensionality and complexity of dependencies
    - k-NN methods generally require fewer samples than KDE
    - Parametric methods (Gaussian) most sample-efficient when assumptions hold

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.information.conditional_mutual_information import conditional_mutual_information
    >>>
    >>> # Generate sample data
    >>> n = 1000
    >>> X = np.random.randn(n, 2)
    >>> Y = np.random.randn(n, 1)
    >>> Z = np.random.randn(n, 1)
    >>>
    >>> # Compute conditional MI using different methods
    >>> cmi_gauss = conditional_mutual_information(X, Y, Z, method='gaussian')
    >>> cmi_knn = conditional_mutual_information(X, Y, Z, method='knn', k=3)
    >>>
    >>> print(f"Gaussian CMI: {cmi_gauss:.3f}")
    >>> print(f"k-NN CMI: {cmi_knn:.3f}")
    """
    if method == "gaussian":
        cmi = gaussian_conditional_mutual_information(X, Y, Z)

    elif method == "kde" or method == "kernel_density":
        cmi = kde_conditional_mutual_information(
            X, Y, Z, bandwidth=bandwidth, kernel=kernel
        )

    elif method == "knn":
        cmi = knn_conditional_mutual_information(
            X, Y, Z, metric=metric, k=k, kd_tree=kd_tree
        )

    elif method == "geometric_knn":
        cmi = geometric_knn_conditional_mutual_information(
            X, Y, Z, metric=metric, k=k, kd_tree=kd_tree
        )

    elif method == "poisson":
        cmi = poisson_conditional_mutual_information(X, Y, Z)

    else:
        supported_methods = [
            "gaussian",
            "kde",
            "kernel_density",
            "knn",
            "geometric_knn",
            "poisson",
        ]
        raise ValueError(
            f"Method '{method}' unavailable. Supported methods: {supported_methods}"
        )

    # Ensure non-negativity: CMI is theoretically always >= 0,
    # but finite sample estimation can produce small negative values.
    # Only clamp finite values; preserve NaN/inf for error handling.
    if np.isfinite(cmi):
        return max(0.0, cmi)
    return cmi
