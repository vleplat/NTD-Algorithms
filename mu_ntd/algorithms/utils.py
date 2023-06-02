import numpy as np
import tensorly as tl
from tensorly.tucker_tensor import tucker_normalize
from tensorly.metrics.factors import congruence_coefficient
import copy

def beta_divergence(a, b, beta):
    """
    Compute the beta-divergence of two floats or arrays a and b,
    as defined in [3].

    TODO: use scikit learn version

    Parameters
    ----------
    a : float or array
        First argument for the beta-divergence.
    b : float or array
        Second argument for the beta-divergence. 
    beta : float
        the beta factor of the beta-divergence.
    
    Returns
    -------
    float
        Beta-divergence of a and b.
        
    References
    ----------
    [1] C. Févotte and J. Idier, Algorithms for nonnegative matrix 
    factorization with the beta-divergence, Neural Computation, 
    vol. 23, no. 9, pp. 2421–2456, 2011.
    """
    if beta < 0:
        #raise err.InvalidArgumentValue("Invalid value for beta: negative one.") from None
        print("Invalid value for beta: negative one.")
    
    if beta == 1:
        return np.sum(a * np.log(a/b, where=(a!=0)) - a + b)
    elif beta == 0:
        return np.sum(a/b - np.log(a/b, where=(a!=0)) - 1)
    else:
        return np.sum(1/(beta*(beta -1)) * (a**beta + (beta - 1) * b**beta - beta * a * (b**(beta-1))))
    

def sparsify(M, s=0.5, epsilon=0):
    """Adds zeroes in matrix M in order to have a ratio s of nnzeroes/nnentries.

    Parameters
    ----------
    M : 2darray
        The input numpy array
    s : float, optional
        the sparsity ratio (0 for fully sparse, 1 for density of the original array), by default 0.5
    """    
    vecM = M.flatten()
    # use quantiles
    val = np.quantile(vecM, 1-s)
    # put zeros in M
    M[M<val]=epsilon
    return M

def tucker_fms(tucker_in, tucker_target):
    """Compute the factor match score for the factors of a Tucker model (nonnegative or sparse for identifiability)
    This is similar to CP fms but allows diverse dimensions on each mode. The formula is :math:

        \prod_fac <fac_in, fac_target>

    Ideally:
    Cores are part of the fms. After permuting factors and core according to factors similarity, the core can be used an another factor.
    In practice:
    Core is discarded.

    Parameters
    ----------
    tucker_in : tucker tensor
        input tucker tensor to compare to the ground truth
    tucker_target : tucker tensor
        ground truth to match

    Return
    ------
    fms, scalar 
    permutations, list of lists
    """
    # normalize
    tucker_in = tucker_normalize(copy.deepcopy(tucker_in))
    tucker_target = tucker_normalize(copy.deepcopy(tucker_target))
    # input processing
    factors_in = tucker_in[1]
    factors_t = tucker_target[1]

    # permute and fms
    fms = 1
    perms = []
    for i in range(len(factors_in)):
        # pad if sizes are different
        diff_col = factors_in[i].shape[1] - factors_t[i].shape[1]
        if diff_col>0:
            factors_t[i] = tl.concatenate((factors_t[i], 1e-16*tl.ones([factors_t[i].shape[0], diff_col])), axis=1)
        elif diff_col<0:
            factors_in[i] = tl.concatenate((factors_in[i], 1e16*tl.ones([factors_in[i].shape[0], diff_col])), axis=1)
        score, perm = congruence_coefficient(factors_in[i], factors_t[i])
        factors_in[i] = factors_in[i][:,perm]
        fms *= score
        perms.append(perm)

    return fms, perms

if __name__== "__main__":
    # test
    import tensorly as tl

    # dummy tucker 1
    tucker1 = tl.tucker_tensor.TuckerTensor((tl.randn([3,2,3]), [tl.randn([5,3]),tl.randn([6,2]), tl.randn([7,3])]))
    #tucker2 = tl.tucker_tensor.TuckerTensor((tl.randn([3,2,3]), [tl.randn([5,3]),tl.randn([6,2]), tl.randn([7,3])]))
    tucker2 = copy.deepcopy(tucker1)
    tucker2[1][0] = tucker2[1][0][:,[0,2,1]] + 0.1*tl.randn([5,3])
    tucker2[0] = tucker2[0][[0,2,1],:,:]
    fms, perms = tucker_fms(tucker1,tucker2)
    print(fms, perms)
