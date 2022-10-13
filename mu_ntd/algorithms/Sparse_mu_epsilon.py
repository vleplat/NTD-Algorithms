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
import itertools as it

def gamma(beta):
    """
    Implements Sparse KL NTD
    """
    if beta<1:
        return 1/(2-beta)
    if beta>2:
        return  1/(beta-1)
    else:
        return 1

def mu_betadivmin(U, V, M, beta, l2weight=0, l1weight=0, epsilon=1e-12, iter_inner=20, acc_alpha=0.5, acc_delta=0.01, atime=1):
    """
    ============================================================
    Sparse Beta-Divergence NMF solved with Multiplicative Update
    ============================================================
    Computes an approximate solution of a beta-NMF
    [3] with ONE STEP of the Multiplicative Update rule [2,3].
    M is m by n, U is m by r, V is r by n.
    All matrices are nonnegative componentwise.
    We are interested in reducing the loss:
            f(U >= 0) = beta_div(M, UV)+1/2 \mu \|U\|_F^2 + 1/2 \lambda \|V\|_1
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
    acc_delta : float, optional
        _description_, by default 0.01

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

    # acceleration init
    res = 1

    # Checks
    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None

    # Precomputations, outside inner loop
    if beta==0:
        a_tilde = l2weight
        c_tilde = 0
        n_mode = 2
        size_mat = U.shape
        args=[]
        for i in range(n_mode):
            args.append(range(0, size_mat[i]))
        # check
        # gradU = np.zeros(np.shape(U))
        
    if beta==1:
        C = np.sum(V.T,axis=0)
    if beta==2:
        VVt = V@V.T
        MVt = M@V.T

    for iters in range(iter_inner):

        # Computing DeltaU
        flag = True # keep track if update is for beta=0
        if beta == 0:
            K = np.dot(U,V)
            if not l2weight:
                denom = np.dot(K**(beta-1),V.T) + l1weight
                deltaU = U * ((np.dot((K**(beta-2) * M),V.T) / denom) ** gamma(beta) -1)
            else:
                # Only case where we don't use the trick because Jeremy is unsure how to implement
                flag = False # no delta U
                K_inverted = K**(-1)
                Ks_inverted = K**(-2)
                B_tilde = (K_inverted)@V.T
                D_tilde = -1*((U)**2)*np.dot((Ks_inverted*M),V.T)
                
                for combination in it.product(*args):
                    x, flag = cubic_roots(a_tilde, B_tilde[combination], c_tilde, D_tilde[combination]) 
                    U[combination] =  np.maximum(x[0],epsilon)
                    #gradU[combination] = (a_tilde)*U[combination]**3+(B_tilde[combination])*U[combination]**2+(c_tilde)*U[combination]+D_tilde[combination]
                #print(tl.norm(gradU))
            
        if beta == 1:
            K = np.dot(U,V)
            # If l2 weight is not used, we default the l1 formula. If l1=0 also it boils down to usual MU. This is faster than the l2 default.
            if not l2weight:
                K_inverted = K**(-1)
                # testing TODO CHECK
                denom = np.array([C for i in range(np.shape(K)[0])]) + l1weight
                deltaU = U * ((np.dot((K_inverted*M),V.T) / denom)-1)
            else:
                #e = np.ones(np.shape(M))
                #C = np.dot(e,V.T)
                K_inverted = K**(-1)
                S = 4*l2weight*U*np.dot((K_inverted*M),V.T)
                denom = 2*l2weight
                deltaU = ((C**2 + S)**(1/2)-C) / denom - U # not so useful here, but uniform syntax
        # TODO: implement beta=2 with l2 --> beta=2 should never be used anyway
        elif beta == 2:
            #deltaU = U * ((MVt / (U@VVt + l1weight))-1)
            # TODO Confirm with Valentin
            deltaU = U * (((MVt - l1weight) / (U@VVt + l2weight*U))-1)
        elif beta == 3:
            K = np.dot(U,V)
            denom = np.dot(K**2,V.T) + l1weight
            deltaU = U * ((np.dot((K * M),V.T) / denom) ** gamma(beta)-1)
        else:
            K = np.dot(U,V)
            denom = np.dot(K**(beta-1),V.T) + l1weight
            deltaU = U * ((np.dot((K**(beta-2) * M),V.T) / denom) ** gamma(beta)-1)

        # Updating U
        if flag:
            U = np.maximum(U + deltaU, epsilon)
            # stopping condition dynamic if allowed
            if acc_delta:
                deltaU_norm = np.linalg.norm(deltaU)
                # if first iteration, store first decrease
                if iters==0:
                    res_0 = deltaU_norm
                else:
                    res = deltaU_norm
                # we stop if deltaV decrease in norm is not enough
                # at least 2 iterations
                if iters>0 and res < acc_delta*res_0:
                    #print("factor, after ", iters, res, res_0) # for debugging
                    break
    return U, iters

def mu_tensorial(G, factors, tensor, beta, l2weight=0, l1weight=0, epsilon=1e-12, iter_inner=20, acc_delta=0.01):
    """
    TODO ACCELERATION

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

    # acceleration init
    res = 1
    res0 = 0

    # Checks
    if beta < 0:
        raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None

    #if not l2weight and not l1weight:
    #    raise err.InvalidArgumentValue("l1 and l2 coefficients may not be nonzero simultaneously for one mode")

    # Precomputations, outside inner loop
    if beta==0:
        a_tilde = l2weight
        c_tilde = 0
        n_mode = len(G.shape)
        size_tens = G.shape
        args=[]
        for i in range(n_mode):
            args.append(range(0, size_tens[i]))
        # check
        #gradG = np.zeros(np.shape(G))
    if beta==1:
        # faster method without creating ones tensor
        sums = [np.sum(fac,axis=0) for fac in factors]
        C = tl.tenalg.outer(sums)
    if beta==2:
        VVt = [fac.T@fac for fac in factors]
        MVt = tl.tenalg.multi_mode_dot(tensor, [fac.T for fac in factors])

    for iters in range(iter_inner):
        
        flag = True
        if beta == 0:
            K = tl.tenalg.multi_mode_dot(G,factors)
            L1 = K**(-1)
            L2 = K**(-2) * tensor
            if not l2weight:
                deltaG = G * ((tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (l1weight +
                tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors]))) ** gamma(beta) -1)
            else:
                flag = False # seems hard to stop on the fly here
                B_tilde = tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors])
                D_tilde = -1*((G)**2)*tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors])
                for combination in it.product(*args):
                    x, flag = cubic_roots(a_tilde, B_tilde[combination], c_tilde, D_tilde[combination]) 
                    G[combination] =  np.maximum(x[0],epsilon)
                    #gradG[combination] = (a_tilde)*G[combination]**3+(B_tilde[combination])*G[combination]**2+(c_tilde)*G[combination]+D_tilde[combination]
                #print(tl.norm(gradG))
        if beta == 1:
            if not l2weight:
                K = tl.tenalg.multi_mode_dot(G,factors)
                L2 = K**(-1) * tensor
                deltaG = G * ((tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (l1weight + C)) - 1)
                
            else:
                K = tl.tenalg.multi_mode_dot(G,factors)
                L2 = K**(-1) * tensor
                S = 4*l2weight*G*tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors])
                denom = 2*l2weight
                deltaG = ((C**2 + S)**(1/2)-C) / denom - G 
 
        elif beta == 2:
            K = tl.tenalg.multi_mode_dot(G,factors)
            #deltaG = G * ((MVt  / (l1weight + tl.tenalg.multi_mode_dot(G, VVt))) - 1)
            # TODO confirm with Valentin
            deltaG = G * (((MVt - l1weight)  / (l2weight*G + tl.tenalg.multi_mode_dot(G, VVt))) - 1)

        elif beta == 3:
            K = tl.tenalg.multi_mode_dot(G,factors)
            L1 = K**2
            L2 = K * tensor
            deltaG = G * ((tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (l1weight +
            tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors]))) ** gamma(beta) -1)

        else:
            K = tl.tenalg.multi_mode_dot(G,factors)
            L1 = K**(beta-1)
            L2 = K**(beta-2) * tensor
            deltaG = G * ((tl.tenalg.multi_mode_dot(L2, [fac.T for fac in factors]) / (l1weight +
            tl.tenalg.multi_mode_dot(L1, [fac.T for fac in factors]))) ** gamma(beta) -1)

        # Updating G
        if flag:
            G = np.maximum(G + deltaG, epsilon)
            # stopping condition dynamic if allowed
            if acc_delta:
                deltaG_norm = np.linalg.norm(deltaG)**2
                # if first iteration, store first decrease
                if iters==0:
                    res_0 = deltaG_norm
                else:
                    res = deltaG_norm
                # we stop if deltaV decrease in norm is not enough
                if iters>0 and res < np.sqrt(acc_delta)*res_0: #TODO note sqrt here empirical
                    #print("core, after ", iters, res, res_0) # for debugging
                    break

    return G, iters

def cubic_roots(a_tilde, b_tilde, c_tilde, d_tilde):
    """
    ============================================================
    Computing the roots of a 3-order polynomial equation
    ============================================================
    Computes the roots of the real positive roots of 
    the polynomial of order 3 in the normal form:
    x^3+px^2+qx+r

    The computation method is inspired by [1].

    Parameters
    ----------
    a_tilde : a positive scalar
        coefficient of the cubic term
    b_tilde : a scalar
        coefficient of the quadratic term
    c_tilde : a scalar
        coefficient of the linear term
    d_tilde : a scalar
        constant term

    Returns
    -------
    x: array
        a 3-by-1 array or a 1-by-1 array containing the roots of x^3+px^2+qx+r
    flag: binary number
        1 if only one real root, o otherwise
    References
    ----------
    [1]: Yan-Bin Jia, Roots of Polynomials,
    Lecture notes, 2020.

    TODO: Vectorize this
    """
    
    # Precomputations of p,q and r
    p = (b_tilde)/a_tilde
    q = c_tilde/a_tilde
    r = (d_tilde)/a_tilde
    
    # Computation of the normal form y^3+ay+b=0 with following CV x=y-p/3
    a = 1/3*(3*q-p**2)
    b = 1/27*(2*p**3-9*p*q+27*r)
    flag = 0
    
    # Computation and discussion based on the radicant
    rad = (b**2)/4+(a**3)/27
    x = []
    
    if rad > 0:
        #print('radicant is positive')
        A = np.cbrt(-b/2+np.sqrt(rad))
        B = np.cbrt(-b/2-np.sqrt(rad))
        x.append(A+B-p/3)
        x.append(-1/2*(A+B)+np.sqrt(3)/2*(A-B)*1j-p/3)
        x.append(-1/2*(A+B)-np.sqrt(3)/2*(A-B)*1j-p/3)
        flag = 1
    elif rad == 0:
        #print('radicant is null')
        if b > 0:
            x.append(-2*np.sqrt(-a/3)-p/3)
            x.append(np.sqrt(-a/3)-p/3)
            x.append(np.sqrt(-a/3)-p/3)
        elif b < 0:
            x.append(2*np.sqrt(-a/3)-p/3)
            x.append(-1*np.sqrt(-a/3)-p/3)
            x.append(-1*np.sqrt(-a/3)-p/3)
        else:
            x.append(0-p/3)
            x.append(0-p/3)
            x.append(0-p/3)
            
    elif rad < 0:
        #print('radicant is negative')
        if b > 0:
            phi = np.arccos(-1*np.sqrt((b**2/4)/(-a**3/27)))
        elif b < 0:
            phi = np.arccos(np.sqrt((b**2/4)/(-a**3/27)))
        else:
            phi = np.arccos(0)

        for i in range(3):
            x.append(2*np.sqrt(-a/3)*np.cos(phi/3+2*i*np.pi/3)-p/3)

        
    return x, flag
