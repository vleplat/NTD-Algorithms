# -*- coding: utf-8 -*-
"""
Created on 2022
@author: vleplat
## Author : Valentin Leplat, based on Florian Voorwinden's code during its internship
#           and Axel Marmoret during his PhD
"""

import numpy as np
import time
import tensorly as tl
import nn_fac.errors as err

def gamma(beta):
    """
    Implements Sparse KL NTD (at the moment only for beta=1)
    """
    if beta<1:
        return 1/(2-beta)
    if beta>2:
        return  1/(beta-1)
    else:
        return 1

def mu_betadivmin(U, V, M, beta, l2weight=0, l1weight=0, epsilon=1e-12):
    """
    ============================================================
    Sparse Beta-Divergence NMF solved with Multiplicative Update
    ============================================================
    Computes an approximate solution of a beta-NMF
    [3] with ONE STEP of the Multiplicative Update rule [2,3].
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.
    We are interested in reducing the loss:
            f(U >= 0) = beta_div(M, UV)+1/2 \mu \|U\|_F^2
    The update rule of this algorithm is inspired by [3].
    Parameters
    ----------
    U : m-by-r array
        The first factor of factorization, the one which will be updated.
    V : r-by-n array
        The second factor of the factorization, which won't be updated.
    M : m-by-n array
        The initial matrix, to approach.
    beta : Nonnegative float
        The beta coefficient for the beta-divergence.
    l2weight : positive float
        The l2 penalty weight. Prevents the norm of factors to go to infinity under l1 regularisation on other factors. Only implemented for beta=1.
        Default: None
    l1weight : positive float
        The l1 penalty weight. Induces sparsity on the factor.
        Default: None
    epsilon: float
        Upper bound on the factors
        Default: 1e-12
    Returns
    -------
    U: array
        a m-by-r nonnegative matrix \approx argmin_{U >= 0} beta_div(M, UV)+1/2 \mu \|U\|_F^2
    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.
    [2] D. Lee and H. S. Seung, Learning the parts of objects by non-negative
    matrix factorization., Nature, vol. 401, no. 6755, pp. 788–791, 1999.
    [3] C. Févotte and J. Idier, Algorithms for nonnegative matrix
    factorization with the beta-divergence, Neural Computation,
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """

    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None


    K = np.dot(U,V)

    if not l2weight and not l1weight:
        raise err.InvalidArgumentValue("l1 and l2 coefficients may not be nonzero simultaneously for one mode")

    if beta == 1:
        # If l2 weight is not used, we default the l1 formula. If l1=0 also it boils down to usual MU. This is faster than the l2 default.
        if not l2weight:
            K_inverted = K**(-1)
            line = np.sum(V.T,axis=0)
            # todo check l1 update formula
            denom = np.array([line for i in range(np.shape(K)[0])]) + l1weight
            return np.maximum(U * (np.dot((K_inverted*M),V.T) / denom),epsilon)
        else:
            e = np.ones(np.shape(M))
            C = np.dot(e,V.T)
            K_inverted = K**(-1)
            S = 4*l2weight*U*np.dot((K_inverted*M),V.T)
            denom = 2*l2weight
            return np.maximum(((C**2 + S)**(1/2)-C) / denom, epsilon)
        # TODO: l1 for other betas
    elif beta == 2:
        denom = np.dot(K,V.T)
        return np.maximum(U * (np.dot(M,V.T) / denom), epsilon)
    elif beta == 3:
        denom = np.dot(K**2,V.T)
        return np.maximum(U * (np.dot((K * M),V.T) / denom) ** gamma(beta), epsilon)
    else:
        denom = np.dot(K**(beta-1),V.T)
        return np.maximum(U * (np.dot((K**(beta-2) * M),V.T) / denom) ** gamma(beta), epsilon)

def mu_tensorial(G, factors, tensor, beta, l2weight=0, l1weight=0, epsilon=1e-12):
    """
    This function is used to update the core G of a
    nonnegative Tucker Decomposition (NTD) [1] with beta-divergence [3]
    and Multiplicative Updates [2] and sparsity penalty.
    Parameters
    ----------
    G : tensorly tensor
        Core tensor at this iteration.
    factors : list of tensorly tensors
        Factors for NTD at this iteration.
    T : tensorly tensor
        The tensor to estimate with NTD.
    beta : Nonnegative float
        The beta coefficient for the beta-divergence.
    l2weight : positive float
        The l2 penalty weight. Prevents the norm of core to go to infinity under l1 regularisation on other factors. Only implemented for beta=1.
    l1weight : positive float
        The l1 penalty weight. Induces sparsity on the core.
    epsilon: float
        Upper bound on the factors
        Default: 1e-12
    Returns
    -------
    G : tensorly tensor
        Update core in NTD.
    References
    ----------
    [1] Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.
    [2] D. Lee and H. S. Seung, Learning the parts of objects by non-negative
    matrix factorization., Nature, vol. 401, no. 6755, pp. 788–791, 1999.
    [3] C. Févotte and J. Idier, Algorithms for nonnegative matrix
    factorization with the beta-divergence, Neural Computation,
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """

    # TODO implement l2 here as well

    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None

    K = tl.tenalg.multi_mode_dot(G,factors)

    if beta == 1:
        L1 = np.ones(np.shape(K))
        L2 = K**(-1) * tensor

    elif beta == 2:
        L1 = K
        L2 = np.ones(np.shape(K)) * tensor

    elif beta == 3:
        L1 = K**2
        L2 = K * tensor

    else:
        L1 = K**(beta-1)
        L2 = K**(beta-2) * tensor

    #return np.maximum(G * (tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (np.ones(np.shape(G))*l1weight + tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors]))) ** gamma(beta) , epsilon)
    return np.maximum(G * (tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (l1weight + tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors]))) ** gamma(beta) , epsilon)
