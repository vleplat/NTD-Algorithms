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

def mu_betadivmin(U, V, M, beta, l2weight=0, l1weight=0, epsilon=1e-12, iter_inner=20):
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
    iter_inner: int
        Number of updates/loops in this call
        Default: 20

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

    # Checks
    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None

    # Precomputations, outside inner loop
    if beta==1:
        C = np.sum(V.T,axis=0)
    if beta==2:
        VVt = V@V.T
        MVt = M@V.T

    for iters in range(iter_inner):

        if beta == 1:
            K = np.dot(U,V)
            # If l2 weight is not used, we default the l1 formula. If l1=0 also it boils down to usual MU. This is faster than the l2 default.
            if not l2weight:
                K_inverted = K**(-1)
                #line = np.sum(V.T,axis=0)
                # todo check l1 update formula
                denom = np.array([C for i in range(np.shape(K)[0])]) + l1weight
                U = np.maximum(U * (np.dot((K_inverted*M),V.T) / denom),epsilon)
            else:
                #e = np.ones(np.shape(M))
                #C = np.dot(e,V.T)
                K_inverted = K**(-1)
                S = 4*l2weight*U*np.dot((K_inverted*M),V.T)
                denom = 2*l2weight
                U = np.maximum(((C**2 + S)**(1/2)-C) / denom, epsilon) # TODO: check broadcasting
            # TODO: l1 for other betas
        elif beta == 2:
            U = np.maximum(U * (MVt / U@VVt), epsilon)
        elif beta == 3:
            K = np.dot(U,V)
            denom = np.dot(K**2,V.T)
            U = np.maximum(U * (np.dot((K * M),V.T) / denom) ** gamma(beta), epsilon)
        else:
            K = np.dot(U,V)
            denom = np.dot(K**(beta-1),V.T)
            U = np.maximum(U * (np.dot((K**(beta-2) * M),V.T) / denom) ** gamma(beta), epsilon)

        # stopping condition dynamic if allowed
        #TODO
        if False:
            print('Stopped inner iters, not implemented')
            break
    return U

def mu_tensorial(G, factors, tensor, beta, l2weight=0, l1weight=0, epsilon=1e-12, iter_inner=20):
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
    iter_inner: int
        Number of updates/loops in this call
        Default: 20

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
    # TODO factorize reusable terms for beta=2    # Checks

    # Checks
    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None

    #if not l2weight and not l1weight:
    #    raise err.InvalidArgumentValue("l1 and l2 coefficients may not be nonzero simultaneously for one mode")

    # Precomputations, outside inner loop
    if beta==1:
        # faster method without creating ones tensor
        sums = [np.sum(fac,axis=0) for fac in factors]
        C = tl.tenalg.outer(sums)
        #C = tl.tenalg.multi_mode_dot(np.ones(np.shape(tensor)), [fac.T for fac in factors])
    if beta==2:
        VVt = [fac@fac.T for fac in factors]
        MVt = tl.tenalg.multi_mode_dot(tensor, [fac.T for fac in factors])

    for iter in range(iter_inner):
        if beta == 1:
            K = tl.tenalg.multi_mode_dot(G,factors)
            L2 = K**(-1) * tensor
            G = np.maximum(G * (tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (l1weight + C)) ** gamma(beta) , epsilon)

        elif beta == 2:
            K = tl.tenalg.multi_mode_dot(G,factors)
            G = np.maximum(G * (MVt  / (l1weight +
            tl.tenalg.multi_mode_dot(G, VVt))) ** gamma(beta) , epsilon)

        elif beta == 3:
            K = tl.tenalg.multi_mode_dot(G,factors)
            L1 = K**2
            L2 = K * tensor
            G = np.maximum(G * (tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (l1weight +
            tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors]))) ** gamma(beta) , epsilon)

        else:
            L1 = K**(beta-1)
            L2 = K**(beta-2) * tensor
            G = np.maximum(G * (tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (l1weight +
            tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors]))) ** gamma(beta) , epsilon)

    #return np.maximum(G * (tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (np.ones(np.shape(G))*l1weight + tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors]))) ** gamma(beta) , epsilon)
    return G