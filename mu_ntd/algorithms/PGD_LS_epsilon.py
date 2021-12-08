# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:04:50 2021

@author: Valentin Leplat
"""

import numpy as np
import time
import tensorly as tl
import mu_ntd.utils.errors as err
import mu_ntd.utils.beta_divergence as beta_div

def PGD_betadivmin(U, V, M, beta, epsilon, lk):
    """
    ===========================================================================
    Beta-Divergence NMF solved with Gradient Descent (with backtracking line-search)
    ===========================================================================

    Computes an approximate solution of a beta-NMF
    [3] with the GD
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.

    The problem is solved for the beta-divergence:

            min_{U >= 0} beta_div(M, UV)



    Parameters
    ----------
    U : m-by-r array
        The first factor of the NNLS, the one which will be updated.
    V : r-by-n array
        The second factor of the NNLS, which won't be updated.
    M : m-by-n array
        The initial matrix, to approach.
    beta : Nonnegative float
        The beta coefficient for the beta-divergence.
    epsilon : lower bound for entries of U and V
    lk: parameter for backtracking line-search

    Returns
    -------
    U: array
        a m-by-r nonnegative matrix \approx argmin_{U >= 0} beta_div(M, UV)

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

    if beta == 1:
        K_inverted = K**(-1)
        line = np.sum(V.T,axis=0)
        gradf_pos = np.array([line for i in range(np.shape(K)[0])])
        gradf_minus = np.dot((K_inverted*M),V.T)

    elif beta == 2:
        gradf_pos = np.dot(K,V.T)
        gradf_minus = np.dot(M,V.T)

    elif beta == 3:
        gradf_pos = np.dot(K**2,V.T)
        gradf_minus = np.dot((K * M),V.T)

    else:
        gradf_pos = np.dot(K**(beta-1),V.T)
        gradf_minus = np.dot((K**(beta-2) * M),V.T)

    # backtracking line-search procedure
    fxk = beta_div.beta_divergence(M, K, beta)
    gradf = (gradf_pos - gradf_minus)
    ik = 0
    while 1:
        T_i = np.maximum(U-1/(lk*2**(ik))*gradf,epsilon)
        f_Ti = beta_div.beta_divergence(M, np.dot(T_i,V), beta)
        rho_lk = fxk + tl.sum(gradf*(T_i - U),axis=None)+(lk*2**(ik))/2*((tl.norm(T_i - U))**2)
        if f_Ti <= rho_lk:
            U = T_i
            lk=(lk*2**(ik))/2
            return U, lk
        else:
            ik = ik+1

def PGD_tensorial(G, factors, tensor, beta, epsilon, lk):
    """
    This function is used to update the core G of a
    nonnegative Tucker Decomposition (NTD) [1] with beta-divergence [3]

    See ntd.py of this module for more details on the NTD (or [1])


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
        epsilon : lower bound for entries of G
    lk: parameter for backtracking line-search

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

    # backtracking line-search procedure
    gradf_minus = tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors])
    gradf_pos = tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors])
    fxk = beta_div.beta_divergence(tensor, tl.tenalg.multi_mode_dot(G, factors), beta)
    gradf = (gradf_pos - gradf_minus)
    ik = 0
    while 1:
        T_i = np.maximum(G-1/(lk*2**(ik))*gradf,epsilon)
        f_Ti = beta_div.beta_divergence(tensor, tl.tenalg.multi_mode_dot(T_i, factors), beta)
        rho_lk = fxk + tl.sum(gradf*(T_i - G),axis=None)+(lk*2**(ik))/2*((tl.norm(T_i - G))**2)
        if f_Ti <= rho_lk:
            G = T_i
            lk=(lk*2**(ik))/2
            return G, lk
        else:
            ik = ik+1
