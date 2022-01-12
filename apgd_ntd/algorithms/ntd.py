# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:52:21 2019
Updated on Wed Dec 29 2021  

@author: amarmore, vleplat
"""

import numpy as np
import scipy
import time
import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker
import math
import mu_ntd.algorithms.nnls as nnls
import mu_ntd.utils.errors as err
import mu_ntd.algorithms.mu_epsilon as mu
import mu_ntd.algorithms.PGD_LS_epsilon as pgd
import mu_ntd.utils.beta_divergence as beta_div
import apgd_ntd.algorithms.APGD_epsilon as apgd

import numpy as np

import scipy.sparse as sci_sparse



#------------------------------------------------------------------------------------------------------------------------------------------------------
# Accelerated Projected Gradient Method with HER for NTD with Frobenius Norm only
#------------------------------------------------------------------------------------------------------------------------------------------------------
def ntd_apgd(tensor, ranks, init = "random", core_0 = None, factors_0 = [], n_iter_max=100, tol=1e-6,
           sparsity_coefficients = [], fixed_modes = [], normalize = [], mode_core_norm = None, beta = 2,
           verbose=False, return_costs=False, deterministic=False, extrapolate=False):
    """
    ======================================
    Nonnegative Tucker Decomposition (NTD)
    ======================================

    Factorization of a tensor T in nonnegative matrices,
    linked by a nonnegative core tensor, of dimensions equal to the ranks
    (in general smaller than the tensor).
    See more details about the NTD in [1].

    For example, in the third-order case, resolution of:
        T \approx (W \otimes H \otimes Q) G

    In this example, W, H and Q are the factors, one per mode, and G is the core tensor.
    W is of size T.shape[0] * ranks[0],
    H is of size T.shape[1] * ranks[1],
    Q is of size T.shape[2] * ranks[2],
    G is of size ranks[0] * ranks[1] * ranks[2].

    More precisely, the chosen optimization algorithm is a projected gradient scheme,
    The projected gradient rule is derived by the authors, and doesn't appear in citation for now.

    Tensors are manipulated with the tensorly toolbox [3].

    In tensorly and in our convention, tensors are unfolded and treated as described in [4].

    Parameters
    ----------
    tensor: tensorly tensor
        The nonnegative tensor T, to factorize
    ranks: list of integers
        The ranks for each factor of the decomposition
    init: "random" | "tucker" | "custom" |
        - If set to random:
            Initializes with random factors of the correct size.
            The randomization is the uniform distribution in [0,1),
            which is the default from numpy random.
        - If set to tucker:
            Resolve a tucker decomposition of the tensor T (by HOSVD) and
            initializes the factors and the core as this resolution, clipped to be nonnegative.
            The tucker decomposition is performed with tensorly [3].
        - If set to "chromas":
            Resolve a tucker decomposition of the tensor T (by HOSVD) and
            initializes the factors and the core as this resolution, clipped to be nonnegative.
            The tucker decomposition is performed with tensorly [3].
            Contrary to "tucker" init, the first factor will then be set to the 12-size identity matrix,
            because it's a decomposition model specific for modeling music expressed in chromas.
        - If set to custom:
            core_0 and factors_0 (see below) will be used for the initialization
        Default: random
    core_0: None or tensor of nonnegative floats
        A custom initialization of the core, used only in "custom" init mode.
        Default: None
    factors_0: None or list of array of nonnegative floats
        A custom initialization of the factors, used only in "custom" init mode.
        Default: None
    n_iter_max: integer
        The maximal number of iteration before stopping the algorithm
        Default: 100
    tol: float
        Threshold on the improvement in objective function value.
        Between two succesive iterations, if the difference between
        both objective function values is below this threshold, the algorithm stops.
        Default: 1e-6
    sparsity_coefficients: list of float (as much as the number of modes + 1 for the core)
        The sparsity coefficients on each factor and on the core respectively.
        If set to None or [], the algorithm is computed without sparsity
        Default: []
    fixed_modes: list of integers (between 0 and the number of modes + 1 for the core)
        Has to be set not to update a factor, taken in the order of modes and lastly on the core.
        Default: []
    normalize: list of boolean (as much as the number of modes + 1 for the core)
        Indicates whether the factors need to be normalized or not.
        The normalization is a l_2 normalization on each of the rank components
        (For the factors, each column will be normalized, ie each atom of the dimension of the current rank).
        Default: []
    mode_core_norm: integer or None
        The mode on which normalize the core, or None if normalization shouldn't be enforced.
        Will only be useful if the last element of the previous "normalise" argument is set to True.
        Indexes of the modes start at 0.
        Default: None
    hals: boolean
        Whether to run hals (true) or gradient (false) update on the core.
        Default (and recommanded): false
    verbose: boolean
        Indicates whether the algorithm prints the monitoring of the convergence
        or not
        Default: False
    return_costs: boolean
        Indicates whether the algorithm should return all objective function
        values and computation time of each iteration or not
        Default: False
    deterministic:
        Runs the algorithm as a deterministic way, by fixing seed in all possible randomisation,
        and optimization techniques in the NNLS, function of the runtime.
        This is made to enhance reproducible research, and should be set to True for computation of results.
    extrapolate:
        Runs the algorithm by using extrapolation techniques HER introduced in [5]

    Returns
    -------
    core: tensorly tensor
        The core tensor linking the factors of the decomposition
    factors: numpy #TODO: For tensorly pulling, replace numpy by backend
        An array containing all the factors computed with the NTD
    cost_fct_vals: list
        A list of the objective function values, for every iteration of the algorithm.
    toc: list, only if return_errors == True
        A list with accumulated time at each iterations

    Example
    -------
    tensor = np.random.rand(80,100,120)
    ranks = [10,20,15]
    core, factors = NTD.ntd(tensor, ranks = ranks, init = "tucker", verbose = True, hals = False,
                            sparsity_coefficients = [None, None, None, None], normalize = [True, True, False, True])

    References
    ----------
    [1] Tamara G Kolda and Brett W Bader. "Tensor decompositions and applications",
    SIAM review 51.3 (2009), pp. 455{500.

    [2]: N. Gillis and F. Glineur, Accelerated Multiplicative Updates and
    Hierarchical ALS Algorithms for Nonnegative Matrix Factorization,
    Neural Computation 24 (4): 1085-1105, 2012.

    [3] J. Kossai et al. "TensorLy: Tensor Learning in Python",
    arxiv preprint (2018)

    [4] Jeremy E Cohen. "About notations in multiway array processing",
    arXiv preprint arXiv:1511.01306, (2015).
    
    [5] A.M.S. Ang and N. Gillis. "Accelerating Nonnegative Matrix Factorization Algorithms Using Extrapolatiog",
    Neural Computation 31 (2): 417-439, 2019.
    """
    factors = []
    nb_modes = len(tensor.shape)

    if type(ranks) is int:
        ranks = [ranks for i in nb_modes]
    elif len(ranks) != nb_modes:
        raise err.InvalidRanksException("The number of ranks is different than the dim of the tensor, which is incorrect.") from None

    for i in range(nb_modes):
        if ranks[i] > tensor.shape[i]:
            #raise err.InvalidRanksException("The " + str(i) + "-th mode rank is larger than the shape of the tensor, which is incorrect.") from None
            ranks[i] = tensor.shape[i]
            #warnings.warn('ignoring MIDI message %s' % msg)
            print("The " + str(i) + "-th mode rank was larger than the shape of the tensor, which is incorrect. Set to the shape of the tensor")

    if init.lower() == "random":
        for mode in range(nb_modes):
            if deterministic:
                seed = np.random.RandomState(mode * 10)
                random_array = seed.rand(tensor.shape[mode], ranks[mode])
            else:
                random_array = np.random.rand(tensor.shape[mode], ranks[mode])
            factors.append(tl.tensor(random_array))

        if deterministic:
            seed = np.random.RandomState(nb_modes * 10)
            core = tl.tensor(seed.rand(np.prod(ranks)).reshape(tuple(ranks)))
        else:
            core = tl.tensor(np.random.rand(np.prod(ranks)).reshape(tuple(ranks)))

        factors = [np.maximum(f, 1e-12) for f in factors]
        core = np.maximum(core, 1e-12)

    elif init.lower() == "tucker":
        if deterministic:
            init_core, init_factors = tl_tucker(tensor, ranks, random_state = 8142)
        else:
            init_core, init_factors = tl_tucker(tensor, ranks)
        factors = [np.maximum(tl.abs(f), 1e-12) for f in init_factors]
        core = np.maximum(tl.abs(init_core), 1e-12)

    elif init.lower() == "custom":
        factors = factors_0
        core = core_0
        if len(factors) != nb_modes:
            raise err.CustomNotEngouhFactors("Custom initialization, but not enough factors")
        else:
            for array in factors:
                if array is None:
                    raise err.CustomNotValidFactors("Custom initialization, but one factor is set to 'None'")
            if core is None:
                raise err.CustomNotValidCore("Custom initialization, but the core is set to 'None'")

    else:
        raise err.InvalidInitializationType('Initialization type not understood: ' + init)


    return compute_ntd_apgd_HER(tensor, ranks, core, factors, n_iter_max=n_iter_max,
                   sparsity_coefficients = sparsity_coefficients, fixed_modes = fixed_modes,
                   normalize = normalize, mode_core_norm = mode_core_norm,
                   verbose=verbose, return_costs=return_costs, beta = beta, deterministic = deterministic, extrapolate = extrapolate)

def compute_ntd_apgd_HER(tensor_in, ranks, core_in, factors_in, n_iter_max=100,
           sparsity_coefficients = [], fixed_modes = [], normalize = [], beta = 2, mode_core_norm=None,
           verbose=False, return_costs=False, deterministic=False, extrapolate=False):

    # initialisation - store the input varaibles
    core = core_in.copy()
    factors = factors_in.copy()
    tensor = tensor_in

    norm_tensor = tl.norm(tensor, 2)

    # set init if problem
    nb_modes = len(tensor.shape)
    if sparsity_coefficients == None or len(sparsity_coefficients) != nb_modes + 1:
        print("Irrelevant number of sparsity coefficient (different from the number of modes + 1 for the core), they have been set to None.")
        sparsity_coefficients = [None for i in range(nb_modes + 1)]
    if fixed_modes == None:
        fixed_modes = []
    if normalize == None or len(normalize) != nb_modes + 1:
        print("Irrelevant number of normalization booleans (different from the number of modes + 1 for the core), they have been set to False.")
        normalize = [False for i in range(nb_modes + 1)]
    if normalize[-1] and (mode_core_norm == None or mode_core_norm < 0 or mode_core_norm >= nb_modes):
        print("The core was asked to be normalized, but an invalid mode was specified. Normalization has been set to False.")
        normalize[-1] = False
    if not normalize[-1] and (mode_core_norm != None and mode_core_norm >= 0 and mode_core_norm < nb_modes):
        print("The core was asked NOT to be normalized, but mode_core_norm was set to a valid norm. Is this a mistake?")

    # initialisation - declare local varaibles
    cost_fct_vals = []     # value of the objective at the "best current" estimates
    cost_fct_vals_fycn=[]  # value of the objective at (factors_y, core_n)
    cost_fct_vals_fycn.append(beta_div.beta_divergence(tensor, tl.tenalg.multi_mode_dot(core, factors), beta))
    cost_fct_vals.append(beta_div.beta_divergence(tensor, tl.tenalg.multi_mode_dot(core, factors), beta))
    tic = time.time()
    toc = []
    alpha_store = []
    epsilon = 0   #1e-12

    # the extrapolation parameters
    if extrapolate:
        alpha=0.5
    else:
        alpha=0
    
    print('Initial Alpha={}'.format(alpha))

    alpha0 = alpha             #extrapolation parameter setting from last improvement
    alphamax = 1               #1 but Andy told us to increase it to have fun, let us see
    alpha_increase = 1.05       #1.1 max
    alpha_reduce = 1/1.5       
    alphamax_increase  = 1.01

    core_n = core.copy()        # non-extrapolated factor estimates
    factors_n = factors.copy()  # non-extrapolated core estimates
    core_y = core.copy()        # extrapolated factor estimates
    factors_y = factors.copy()  # extrapolated core estimates

    # Iterate over one step of NTD
    for iteration in range(n_iter_max):

        # One pass of APGD on each updated mode
        core, factors, core_n, factors_n, core_y, factors_y, cost, cost_fycn, alpha, alpha0, alphamax = one_ntd_step_apgd_HER(tensor, ranks, core, factors, core_n, factors_n, core_y, factors_y, beta, norm_tensor,
                                              fixed_modes, normalize, mode_core_norm, alpha, cost_fct_vals_fycn, epsilon, alpha0, alphamax, alpha_increase, alpha_reduce, alphamax_increase, cost_fct_vals )

        # Store the computation time, obj value, alpha
        toc.append(time.time() - tic)
        cost_fct_vals.append(cost)
        cost_fct_vals_fycn.append(cost_fycn)
        alpha_store.append(alpha)

        if verbose:
            if iteration == 0:
                print('Initial Obj={}'.format(cost))
            else:
                if cost_fct_vals[-2] - cost_fct_vals[-1] > 0:
                    print('Iter={}|Obj={}| Var={}.'.format(iteration,
                            cost_fct_vals[-1], (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2]))))
                else:
                    # print in red when the reconstruction error is negative (shouldn't happen)
                    print('\033[91m' + 'Iter={}|Obj={}| Var={}.'.format(iteration,
                            cost_fct_vals[-1], (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2]))) + '\033[0m')


    if return_costs:
        return core, factors, cost_fct_vals, toc, alpha_store
    else:
        return core, factors

def one_ntd_step_apgd_HER(tensor, ranks, in_core, in_factors, in_core_n, in_factors_n, in_core_y, in_factors_y, beta, norm_tensor,
                   fixed_modes, normalize, mode_core_norm, alpha, cost_fct_vals_fycn, epsilon, alpha0, alphamax, alpha_increase, alpha_reduce, alphamax_increase, cost_fct_vals):
    # Copy
    core = in_core.copy()
    factors = in_factors.copy()
    core_n = in_core_n.copy()
    core_n_up = core_n.copy()
    factors_n = in_factors_n.copy()
    factors_n_up = factors_n.copy()
    core_y = in_core_y.copy()
    factors_y = in_factors_y.copy()

    cost_fct_val = cost_fct_vals[-1]

    # Store the value of the objective (loss) function at the current
    # iterate (factors_y, core_n).
    cost0_fct_vals_fycn = cost_fct_vals_fycn[-1]

    # Generating the mode update sequence
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]
    
    # Generating list of W.T*W, H.T*H, etc matrices for Lipschitz constant of gradient
    # prod_factors_y=[]
    # Alternative way for computing Lipschitz constant of gradient - Property of Kronecker products
    sigma_factors_y=[]
    # Compute the extrapolated update for the factors.
    # Note that when alpha is zero, factors_y = factors_n.
    for mode in modes_list:
        factors_n_up[mode] = apgd.APGD_factors(factors_y[mode], tl.unfold(tl.tenalg.multi_mode_dot(core_y, factors_y, skip = mode), mode), tl.unfold(tensor,mode), beta, epsilon)
        factors_y[mode] = factors_n_up[mode]+alpha*(factors_n_up[mode]-in_factors_n[mode])
        # prod_factors_y.append(np.dot(factors_y[mode].T,factors_y[mode]))
        u, s, vh = np.linalg.svd(factors_y[mode], full_matrices=False)
        sigma_factors_y.append(s.max()**2);
    # Compute the extrapolated update for the core.
    # Note that when alpha is zero, core_y = core_n.
    core_n_up = apgd.APGD_tensorial(core_y, factors_y, tensor, beta, epsilon, sigma_factors_y)
    core_y = core_n+alpha*(core_n-in_core_n)

    # Compute the value of the objective (loss) function at the
    # extrapolated solution for the factors (factors_y) and the
    # non-extrapolated solution for the core (core_n).
    cost_fycn = beta_div.beta_divergence(tensor, tl.tenalg.multi_mode_dot(core_n, factors_y), beta)

    # Update the extrapolation parameters following Algorithm 3 of
    # Ang & Gillis (2019).
    if(cost_fycn >=cost0_fct_vals_fycn):
        # The solution did not improve, so restart the extrapolation
        # scheme.
        factors_y = in_factors_n.copy()
        core_y = in_core_n.copy()
        alphamax = alpha0
        alpha = alpha_reduce*alpha
    else:
        # The solution improved; retain the basic co-ordinate ascent
        # update as well.
        factors_n = factors_n_up.copy()
        core_n = core_n_up.copy()
        alpha = np.minimum(alphamax,alpha*alpha_increase)
        alpha0 = alpha
        alphamax = np.minimum(1,alphamax_increase*alphamax)

    # If the solution improves the "current best" estimate, update the
    # current best estimate using the non-extrapolated estimates of the
    # core (core_n) and the extrapolated estimates of the factors (factors_y).
    if(cost_fycn < cost_fct_vals[-1]):
        factors = factors_y.copy()
        core = core_n_up.copy()
        cost_fct_val = cost_fycn


    return core, factors, core_n, factors_n, core_y, factors_y, cost_fct_val, cost_fycn, alpha, alpha0, alphamax



