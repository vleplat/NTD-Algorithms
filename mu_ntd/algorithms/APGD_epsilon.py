# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:04:50 2021

@author: Valentin Leplat
"""

import numpy as np
import tensorly as tl

def APGD_factors(U, V, M, beta, epsilon):
    """
    ===========================================================================
    Frobenius NMF solved with Gradient Descent
    ===========================================================================

    Computes an approximate solution of a Frob-NMF
    with the gradient descent
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.

    The problem is solved for the beta-divergence:

            min_{U >= 0} ||M - UV||_F^2

    Parameters
    ----------
    U : m-by-r array
        The first factor of the NNLS, the one which will be updated.
    V : r-by-n array
        The second factor of the NNLS, which won't be updated.
    M : m-by-n array
        The initial matrix, to approach.
    epsilon : lower bound for entries of U and V

    Returns
    -------
    U: array
        a m-by-r nonnegative matrix \approx argmin_{U >= 0} ||M - UV||_F^2

    References
    ----------
    [1]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [2] D. Lee and H. S. Seung, Learning the parts of objects by non-negative
    matrix factorization., Nature, vol. 401, no. 6755, pp. 788–791, 1999.
    """
    # Faster but inaccurate L computation (over-approximation)
    #L = tl.norm(V, 2)**2
    # Slower but accurate L computation
    u, s, vh = np.linalg.svd(V, full_matrices=False)
    L = np.linalg.norm(s**2)
    K = np.dot(U,V)
    gradf_pos = np.dot(K,V.T)
    gradf_minus = np.dot(M,V.T)
    
    gradf = (gradf_pos - gradf_minus)
    U = np.maximum(U-1/(L)*gradf,epsilon)
    
    return U

def APGD_tensorial(G, factors, tensor, beta, epsilon, sigma_factors):
    """
    This function is used to update the core G of a
    nonnegative Tucker Decomposition (NTD) [1] with Frobenius norm 

    See ntd.py of this module for more details on the NTD (or [1])


    Parameters
    ----------
    G : tensorly tensor
        Core tensor at this iteration.
    factors : list of tensorly tensors
        Factors for NTD at this iteration.
    T : tensorly tensor
        The tensor to estimate with NTD.

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
    """
    # computation of Lipschitz constant for gradient - exact for Frob but slow - replace sigma_factors by prod_factors in inputs
    # UtU=tl.tenalg.kronecker(prod_factors, skip_matrix=None, reverse=False)
    # L = np.linalg.norm(UtU,ord='fro')
    # computation of Lipschitz constant for gradient - inexact for Frob but fast and exact for l2 norm
    L = np.prod(sigma_factors)
    
    K = tl.tenalg.multi_mode_dot(G,factors)
    L1 = K
    L2 = np.ones(np.shape(K)) * tensor
    gradf_minus = tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors])
    gradf_pos = tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors])
    gradf = (gradf_pos - gradf_minus)
    G = np.maximum(G-1/(L)*gradf,epsilon)
    return G

   