# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:52:21 2019

@author: veplat, based on amarmoret code
"""

import numpy as np
import time
from tqdm import tqdm
import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker

import mu_ntd.algorithms.Sparse_mu_epsilon as mu
import nn_fac.errors as err
import nn_fac.beta_divergence as beta_div

import numpy as np


######################### Temporary, to test mu and not break everything
def sntd_mu(tensor, ranks, l2weights=None, l1weights=None, init = "random", core_0 = None, factors_0 = [], n_iter_max=100, tol=1e-6,
           fixed_modes = [], beta = 2, accelerate=True,
           verbose=False, return_costs=False, deterministic=False, extrapolate=False, epsilon=1e-12, iter_inner=20):
    """
    ======================================
    Sparse Nonnegative Tucker Decomposition (sNTD) with beta-div loss
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

    More precisely, the chosen optimization algorithm is a multiplicative update scheme, with a beta-divergence loss elementwise. By default beta=1 which corresponds to the Kullback-Leibler divergence.
    The MU rule is derived by the authors, and doesn't appear in citation for now.

    Sparsity is handled using a l1 penalty term on the sparse factors, and a l2 ridge penalty on other factors to avoid scaling degeneracy.

    Tensors are manipulated with the tensorly toolbox [3]. In tensorly and in our convention, tensors are unfolded and treated as described in [4].

    Parameters
    ----------
    tensor: tensorly tensor
        The nonnegative tensor T, to factorize
    ranks: list of integers
        The ranks for each factor of the decomposition
    l2weights: list of floats
        The regularisation parameters using Euclidean norm as a penalisation.
        Prevents scaling degeneracy. Use on all factors which do not have l1
        regularisation, if l1 is used.
    l1weights: list of floats
        The regularisation parameters using l1 norm as a penalisation. Induces sparsity. 
        Penalties are [factor_1,..., factor_n, core]
        Default: None
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
        Between two successive iterations, if the difference between
        both objective function values is below this threshold, the algorithm stops.
        Default: 1e-6
    fixed_modes: list of integers (between 0 and the number of modes + 1 for the core)
        Has to be set not to update a factor, taken in the order of modes and lastly on the core.
        Default: []
    verbose: boolean
        Indicates whether the algorithm prints the monitoring of the convergence
        or not
        Default: False
    beta : float
        The value of the beta parameter in the beta-divergence loss.
        Default : 1
    accelerate: boolean or list of length 2
        If True, inner iterations in the algorithm will stop dynamically based on the rules in [2].
        If False, the number of inner iterations is fixed. In practice use accelerate=True, but
        for fair comparison with other methods accelerate=False has merit. 
        User can input value for the delta parameter by setting accelerate=delta, e.g.
        accelerate = 0.01 is the default value.
        Default: True
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
    epsilon: float
        Lower-bound for the values in factors for MU.
        Default: 1e-12
    iter_inner: int
        Number of inner iterations for each factor/core update.
        Default: 20

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
    alpha_store: list
        A list with all extrapolation steps alpha stored during the iterations, for debugging.

    Example
    -------
    TODO

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

    # Processing l1 and l2 coefficients
    if l1weights is None:
        l1weights = [None for i in range(nb_modes + 1)]
    if type(l1weights)==int or type(l1weights)==float:
        # allow single value input
        l1weights = [l1weights]*(nb_modes+1)
    if len(l1weights) != nb_modes + 1:
        print("Irrelevant number of l1weights coefficient (different from the number of modes + 1 for the core), they have been set to None.")
        l1weights = [None for i in range(nb_modes + 1)]
    
    if l2weights is None:
        l2weights = [None for i in range(nb_modes + 1)]
    if type(l2weights)==int or type(l2weights)==float:
        # allow single value input
        l2weights = [l2weights]*(nb_modes+1)
    if len(l2weights) != nb_modes + 1:
        print("Irrelevant number of l2weights coefficient (different from the number of modes + 1 for the core), they have been set to None.")
        l2weights = [None for i in range(nb_modes + 1)]

    # Checking if l1 and l2 regularisation has been used on the same mode.
    # l1 and l2 being true is the problem
    truthtable = [l1weights[i] and l2weights[i] for i in range(len(l1weights))]
    if any(truthtable):
        raise err.InvalidArgumentValue("A l2 and l1 regularization have been imposed on the same mode, which is not supported at the moment")

    # A bunch of checks for the inputs
    #TODO: set them as warnings
    nb_modes = len(tensor.shape)
    if fixed_modes == None:
        fixed_modes = []

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

        # Avoid zeroes in init, otherwise zero-locking phenomenon
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

    return compute_sntd_mu_HER(tensor, ranks, l2weights, l1weights, core, factors, n_iter_max=n_iter_max,
                       fixed_modes = fixed_modes, accelerate=accelerate,
                       verbose=verbose, return_costs=return_costs, beta = beta, epsilon=epsilon, extrapolate=extrapolate, iter_inner=iter_inner)


def compute_sntd_mu_HER(tensor_in, ranks, l2weights, l1weights, core_in, factors_in, n_iter_max=100,
           fixed_modes = [], beta = 2, accelerate=True,
           verbose=False, return_costs=False, epsilon=1e-12, extrapolate=False, iter_inner=50):

    # initialisation - store the input varaibles
    core = core_in.copy()
    factors = factors_in.copy()
    tensor = tensor_in

    norm_tensor = tl.norm(tensor, 2)

    # initialisation - declare local varaibles
    cost_fct_vals = []     # value of the objective at the "best current" estimates
    cost_fct_vals_fycn=[]  # value of the objective at (factors_y, core_n)
    cost_fct_vals_fycn.append(beta_div.beta_divergence(tensor, tl.tenalg.multi_mode_dot(core, factors), beta))
    cost_fct_vals.append(cost_fct_vals_fycn[0])
    tic = time.time()
    toc = [0]
    alpha_store = []
    inner_cnt = []

    # the acceleration parameters
    if accelerate:
        if type(accelerate)==bool:
            acc_delta = 0.5
        else:
            acc_delta = accelerate
    else:
        acc_delta = 0
    # storing for iters>0
    acc_delta_store = acc_delta

    # the extrapolation parameters
    if extrapolate:
        alpha=0.05
        if verbose:
            print('Initial Alpha={}'.format(alpha))

        alpha0 = alpha             #extrapolation parameter setting from last improvement
        alphamax = 1               #1 but Andy told us to increase it to have fun, let us see
        alpha_increase = 1.1       #1.1 max
        alpha_reduce = 0.2         #0.75 in andersen
        alphamax_increase  = 1.05
    else:
        # no extrapolation
        alpha=0
        if verbose:
            print('Initial Alpha={}'.format(alpha))

        alpha0 = alpha             #extrapolation parameter setting from last improvement
        alphamax = 0               #1 but Andy told us to increase it to have fun, let us see
        alpha_increase = 1       #1.1 max
        alpha_reduce = 1         #0.75
        alphamax_increase  = 1
    alpha_store.append(alpha)

    core_n = core.copy()        # non-extrapolated factor best estimates
    factors_n = factors.copy()  # non-extrapolated core best estimates
    core_y = core.copy()        # extrapolated factor side estimates
    factors_y = factors.copy()  # extrapolated core side estimates

    # Iterate over one step of NTD
    for iteration in tqdm(range(n_iter_max)):

        # For first iteration, no acceleration since refinement is much stronger than with other iterations
        if iteration==0:
            acc_delta = 0
        else:
            acc_delta = acc_delta_store

        # One pass of MU on each updated mode
        core, factors, core_n, factors_n, core_y, factors_y, cost, cost_fycn, alpha, alpha0, alphamax, cnt = one_sntd_step_mu_HER(tensor, ranks, l2weights=l2weights, l1weights=l1weights, in_core=core, in_factors=factors, in_core_n=core_n, in_factors_n=factors_n, in_core_y=core_y, in_factors_y=factors_y, beta=beta, norm_tensor=norm_tensor, fixed_modes=fixed_modes, alpha=alpha, cost_fct_vals_fycn=cost_fct_vals_fycn, epsilon=epsilon, alpha0=alpha0, alphamax=alphamax, alpha_increase=alpha_increase, alpha_reduce=alpha_reduce, alphamax_increase=alphamax_increase, cost_fct_vals=cost_fct_vals, iter_inner=iter_inner, acc_delta=acc_delta)

        # Store the computation time, obj value, alpha, inner iter count
        toc.append(time.time() - tic)
        cost_fct_vals.append(cost)
        cost_fct_vals_fycn.append(cost_fycn)
        alpha_store.append(alpha)
        [inner_cnt.append(elem) for elem in cnt]

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

        # if iteration > 0 and (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2])) < tol:
        #     # Stop condition: relative error between last two iterations < tol
        #     if verbose:
        #         print('Converged to the required tolerance in {} iterations.'.format(iteration))
        #     break

    if return_costs:
        return core, factors, cost_fct_vals, toc, alpha_store, inner_cnt
    else:
        return core, factors

def one_sntd_step_mu_HER(tensor, ranks, l2weights=0, l1weights=0, in_core=0, in_factors=0, in_core_n=0, in_factors_n=0, in_core_y=0, in_factors_y=0, beta=2, norm_tensor=1,
                   fixed_modes=[], alpha=0, cost_fct_vals_fycn=0, epsilon=1e-12, alpha0=0, alphamax=0, alpha_increase=0, alpha_reduce=0, alphamax_increase=0, cost_fct_vals=0, iter_inner=50, acc_delta=0.5):
    # No Copy
    core = in_core#.copy()
    factors = in_factors#.copy()
    core_n = in_core_n#.copy()
    core_n_up = core_n#.copy()
    factors_n = in_factors_n#.copy()
    factors_n_up = factors_n#.copy()
    core_y = in_core_y#.copy()
    factors_y = in_factors_y#.copy()

    cost_fct_val = cost_fct_vals[-1]

    # Store the value of the objective (loss) function at the current
    # iterate (factors_y, core_n).
    cost0_fct_vals_fycn = cost_fct_vals_fycn[-1]

    # Storing the inner iterations count
    inner_cnt = []

    # Generating the mode update sequence
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    # Compute the extrapolated update for the factors.
    # Note that when alpha is zero, factors_y = factors_n.
    for mode in modes_list:
        factors_n_up[mode], cnt = mu.mu_betadivmin(factors_y[mode], tl.unfold(tl.tenalg.multi_mode_dot(core_y, factors_y, skip = mode), mode),
            tl.unfold(tensor,mode), beta, l2weight=l2weights[mode], l1weight=l1weights[mode], epsilon=epsilon, iter_inner=iter_inner,
            acc_delta=acc_delta)
        factors_y[mode] = np.maximum(factors_n_up[mode]+alpha*(factors_n_up[mode]-in_factors_n[mode]),epsilon)
        inner_cnt.append(cnt)
    # Compute the extrapolated update for the core.
    # Note that when alpha is zero, core_y = core_n.
    core_n_up, cnt = mu.mu_tensorial(core_y, factors_y, tensor, beta, l2weight=l2weights[-1], l1weight=l1weights[-1],
                                 epsilon=epsilon, iter_inner=iter_inner, acc_delta=acc_delta)
    core_y = np.maximum(core_n_up+alpha*(core_n_up-in_core_n), epsilon) #TODO check bug correction core_n_up?
    inner_cnt.append(cnt)

    # Compute the value of the objective (loss) function at the
    # extrapolated solution for the factors (factors_y) and the
    # non-extrapolated solution for the core (core_n).
    # ---> No, we only did that for fast computation. Here there is no such fast comp, so we do it on the true estimates
    # TODO: discuss OK
    cost_fycn = beta_div.beta_divergence(tensor, tl.tenalg.multi_mode_dot(core_n_up, factors_n_up), beta)+ 1/2*l2weights[-1]*tl.norm(core_n_up)**2 + l1weights[-1]*tl.sum(core_n_up)
    for mode in modes_list:
        cost_fycn = cost_fycn + 1/2*l2weights[mode]*tl.norm(factors_n_up[mode])**2 + l1weights[mode]*tl.sum(factors_n_up[mode])

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
        #alphamax = np.minimum(0.99,alphamax_increase*alphamax)
        alphamax = np.minimum(2,alphamax_increase*alphamax) # hard cap at 2

    # If the solution improves the "current best" estimate, update the
    # current best estimate using the non-extrapolated estimates of the
    # core (core_n) and the extrapolated estimates of the factors (factors_y).
    if(cost_fycn < cost_fct_vals[-1]):
        factors = factors_y.copy()
        core = core_n_up.copy()
        cost_fct_val = cost_fycn


    return core, factors, core_n, factors_n, core_y, factors_y, cost_fct_val, cost_fycn, alpha, alpha0, alphamax, inner_cnt

#def compute_sntd_mu(tensor_in, ranks, l2weights, l1weights, core_in, factors_in, n_iter_max=100, tol=1e-6,
           #fixed_modes = [], normalize = [], beta = 2, mode_core_norm=None,
           #verbose=False, return_costs=False, deterministic=False):

    ## initialisation - store the input varaibles
    #core = core_in.copy()
    #factors = factors_in.copy()
    #tensor = tensor_in

    #norm_tensor = tl.norm(tensor, 2)

    ## initialisation - declare local varaibles
    #cost_fct_vals = []
    #tic = time.time()
    #toc = []
    #epsilon = 1e-12
    ## initialisation - unfold the tensor according to the modes
    ##unfolded_tensors = []
    ##for mode in range(tl.ndim(tensor)):
    ##   unfolded_tensors.append(tl.base.unfold(tensor, mode))

    ## Iterate over one step of NTD
    #for iteration in range(n_iter_max):
        ## One pass of least squares on each updated mode
        #core, factors, cost = one_sntd_step_mu(tensor, ranks, l2weights, core, factors, beta, norm_tensor,
                                              #fixed_modes, normalize, mode_core_norm,epsilon)

        ## Store the computation time
        #toc.append(time.time() - tic)

        #cost_fct_vals.append(cost)

        #if verbose:
            #if iteration == 0:
                #print('Initial Obj={}'.format(cost))
            #else:
                #if cost_fct_vals[-2] - cost_fct_vals[-1] > 0:
                    #print('Iter={}|Obj={}| Var={} (target is {}).'.format(iteration,
                            #cost_fct_vals[-1], (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2])),tol))
                #else:
                    ## print in red when the reconstruction error is negative (shouldn't happen)
                    #print('\033[91m' + 'Iter={}|Obj={}| Var={} (target is {}).'.format(iteration,
                            #cost_fct_vals[-1], (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2])),tol) + '\033[0m')

        #if iteration > 0 and (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2])) < tol:
            ## Stop condition: relative error between last two iterations < tol
            #if verbose:
                #print('Converged to the required tolerance in {} iterations.'.format(iteration))
            #break

    #if return_costs:
        #return core, factors, cost_fct_vals, toc
    #else:
        #return core, factors

#def one_sntd_step_mu(tensor, ranks, l2weights, in_core, in_factors, beta, norm_tensor,
                   #fixed_modes, normalize, mode_core_norm, epsilon):
    ## Copy
    #core = in_core.copy()
    #factors = in_factors.copy()

    ## Generating the mode update sequence
    #modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    #for mode in modes_list:
        #factors[mode] = mu.mu_betadivmin(factors[mode], tl.unfold(tl.tenalg.multi_mode_dot(core, factors, skip = mode), mode), tl.unfold(tensor,mode), beta, l2weights[mode+1])

    #core = mu.mu_tensorial(core, factors, tensor, beta, l2weights[0])

    #if normalize[-1]:
        #unfolded_core = tl.unfold(core, mode_core_norm)
        #for idx_mat in range(unfolded_core.shape[0]):
            #if tl.norm(unfolded_core[idx_mat]) != 0:
                #unfolded_core[idx_mat] = unfolded_core[idx_mat] / tl.norm(unfolded_core[idx_mat], 2)
        #core = tl.fold(unfolded_core, mode_core_norm, core.shape)


    #reconstructed_tensor = tl.tenalg.multi_mode_dot(core, factors)

    #cost_fct_val = beta_div.beta_divergence(tensor, reconstructed_tensor, beta)+l2weights[0]*tl.norm(core, order=1)
    #for mode in modes_list:
        #cost_fct_val = cost_fct_val+1/2*l2weights[mode+1]*np.linalg.norm(factors[mode],'fro')**2

    #return core, factors, cost_fct_val #  exhaustive_rec_error








