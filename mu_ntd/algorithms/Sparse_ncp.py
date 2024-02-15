# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 2023

@author: veplat and cohenjer, based on amarmoret code
"""

import numpy as np
import time
from tqdm import tqdm
import copy
import tensorly as tl
from tensorly.cp_tensor import CPTensor
from tensorly.decomposition import tucker as tl_tucker

from tensorly.solvers.penalizations import (
    process_regularization_weights,
    cp_opt_balance
)
import mu_ntd.algorithms.Sparse_mu_epsilon as mu
from mu_ntd.algorithms.utils import beta_divergence as beta_div


######################### Temporary, to test mu and not break everything
def sncp_mu(tensor, rank, l2weights=None, l1weights=None, init = "random", factors_0 = [], n_iter_max=100,
           fixed_modes = [], beta = 2, accelerate=True,
           verbose=False, return_costs=False, epsilon=1e-12, iter_inner=20, opt_rescale=False, print_it=10):
    """
    ======================================
    Sparse Nonnegative Canonical Polyadic Decomposition (sNCP) with beta-div loss
    ======================================

    Factorization of a tensor T in a sum of rank one nonnegative tensors.
    See more details about the NCP in [TODO]

    For example, in the third-order case, resolution of:
        T \approx (W \otimes H \otimes Q) I_r

    In this example, W, H and Q are the factors, one per mode.
    W is of size T.shape[0] * r,
    H is of size T.shape[1] * r,
    Q is of size T.shape[2] * r,

    More precisely, the chosen optimization algorithm is a multiplicative update scheme, with a beta-divergence loss elementwise. By default beta=1 which corresponds to the Kullback-Leibler divergence.
    The MU rule is derived by the authors, and doesn't appear in citation for now.

    Sparsity is handled using a l1 penalty term on the sparse factors, and a l2 ridge penalty on other factors to avoid scaling degeneracy.

    Tensors are manipulated with the tensorly toolbox [3]. In tensorly and in our convention, tensors are unfolded and treated as described in [4].

    Parameters
    ----------
    tensor: tensorly tensor
        The nonnegative tensor T, to factorize
    rank: integer
        The rank for the decomposition
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
            initializes the factors as this resolution, clipped to be nonnegative.
            The tucker decomposition is performed with tensorly [3].
        - If set to custom:
            factors_0 (see below) will be used for the initialization TODO: use CP input directly
        Default: random
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
    epsilon: float
        Lower-bound for the values in factors for MU.
        Default: 1e-12
    iter_inner: int
        Number of inner iterations for each factor/core update.
        Default: 20
    opt_rescale: boolean or string
        Determines if the optimal rescaling is used to minimize regularization terms with respect to factors scale. Use "optimal" for the tensor adaptive sinkhorn strategy, or "scalar" for the non-separable closed-form heuristic. 
        Disabled if one of the factors has no regularization.
        Default: False
    Returns
    -------
    CPTensor:
        An array containing all the factors computed with the sNCP
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

    # TODO: rename weights
    (
        l2weights,
        l1weights,
        disable_rebalance,
        hom_deg,
    ) = process_regularization_weights(
        ridge_coefficients=l2weights,
        sparsity_coefficients=l1weights,
        n_modes=nb_modes,
        rescale=opt_rescale,
        pop_l2=False
    )

    # A bunch of checks for the inputs
    #TODO: set them as warnings
    nb_modes = len(tensor.shape)
    if fixed_modes == None:
        fixed_modes = []

    if init.lower() == "random":
        for mode in range(nb_modes):
            factors.append(np.random.rand(tensor.shape[mode], rank))

        # Avoid zeroes in init, otherwise zero-locking phenomenon
        factors = [np.maximum(f, epsilon) for f in factors]

    elif init.lower() == "tucker":
        _, init_factors = tl_tucker(tensor, nb_modes*[rank])
        factors = [np.maximum(tl.abs(f), epsilon) for f in init_factors]

    elif init.lower() == "custom":
        factors = copy.deepcopy(factors_0)
        if len(factors) != nb_modes:
            #raise err.CustomNotEngouhFactors("Custom initialization, but not enough factors")
            print("Custom initialization, but not enough factors") # TODO improve
        else:
            for array in factors:
                if array is None:
                    #raise err.CustomNotValidFactors("Custom initialization, but one factor is set to 'None'")
                    print("Custom initialization, but one factor is set to 'None'") # TODO

    else:
        #raise err.InvalidInitializationType('Initialization type not understood: ' + init)
        print('Initialization type not understood: ' + init) # TODO

    return compute_sncp_mu_HER(tensor, l2weights, l1weights, factors, n_iter_max=n_iter_max,
                       fixed_modes = fixed_modes, accelerate=accelerate,
                       verbose=verbose, return_costs=return_costs, beta = beta, epsilon=epsilon, iter_inner=iter_inner, opt_rescale=opt_rescale, disable_rebalance=disable_rebalance, hom_deg=hom_deg, print_it=print_it)


def compute_sncp_mu_HER(tensor_in, l2weights, l1weights, factors_in, n_iter_max=100,
           fixed_modes = [], beta = 2, accelerate=True,
           verbose=False, return_costs=False, epsilon=1e-12, iter_inner=50, opt_rescale=False, disable_rebalance=True, hom_deg=None, print_it=10):

    # initialisation - store the input varaibles
    factors = factors_in.copy()
    tensor = tensor_in
    ndims = tl.ndim(tensor)

    # initialisation - declare local varaibles
    cost_fct_vals = []     # value of the objective at the "best current" estimates
    loss = beta_div(tensor, tl.cp_tensor.cp_to_tensor((None,factors)), beta)
    regs_facs = [l1weights[i]*tl.sum(tl.abs(factors[i])) + l2weights[i]*tl.sum(factors[i]**2) for i in range(ndims)]
    cost_fct_vals.append(loss + sum(regs_facs))
    tic = time.time()
    toc = [0]
    alpha_store = []
    inner_cnt = []

    # Computing sparsity
    sparsity = [[0] for i in range(len(tensor.shape))]
    for i in range(len(tensor.shape)):
        sparsity[i]= [(np.sum(factors[i]>(epsilon*2))/np.prod(np.shape(factors[i])))]

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

    # Iterate over one step of NTD
    for iteration in tqdm(range(n_iter_max)):

        # For first iteration, no acceleration since refinement is much stronger than with other iterations
        if iteration==0:
            acc_delta = 0
        else:
            acc_delta = acc_delta_store

        # One pass of MU on each updated mode
        factors, cost, cnt = one_sncp_step_mu(tensor, l2weights=l2weights, l1weights=l1weights, factors=factors, beta=beta, fixed_modes=fixed_modes, epsilon=epsilon, iter_inner=iter_inner, acc_delta=acc_delta, opt_rescale=opt_rescale, disable_rebalance=disable_rebalance, hom_deg=hom_deg)

        # Store the computation time, obj value, alpha, inner iter count
        toc.append(time.time() - tic)
        cost_fct_vals.append(cost)
        [inner_cnt.append(elem) for elem in cnt]

        # Computing sparsity
        for i in range(len(tensor.shape)):
            sparsity[i].append(np.sum(factors[i]>(epsilon*2))/np.prod(np.shape(factors[i])))

        if verbose:
            if iteration == 0:
                print('Initial Obj={}'.format(cost))
            else:
                if cost_fct_vals[-2] - cost_fct_vals[-1] > 0:
                    if not iteration % print_it:
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
        return CPTensor((None,factors)), cost_fct_vals, toc, alpha_store, inner_cnt, sparsity
    else:
        return CPTensor((None,factors))

def one_sncp_step_mu(tensor, l2weights=0, l1weights=0, factors=0, beta=2,
                   fixed_modes=[], epsilon=1e-12, iter_inner=50, acc_delta=0.5, opt_rescale=False, disable_rebalance=True, hom_deg=None):
    """todo

    Parameters
    ----------
    tensor : _type_
        _description_
    l2weights : int, optional
        _description_, by default 0
    l1weights : int, optional
        _description_, by default 0
    factors : int, optional
        _description_, by default 0
    beta : int, optional
        _description_, by default 2
    fixed_modes : list, optional
        _description_, by default []
    epsilon : _type_, optional
        _description_, by default 1e-12
    iter_inner : int, optional
        _description_, by default 50
    acc_delta : float, optional
        _description_, by default 0.5
    opt_rescale : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    
    ndims = tl.ndim(tensor)
    rank = factors[0].shape[1]

    # Storing the inner iterations count
    inner_cnt = []

    # Generating the mode update sequence
    modes_list = [mode for mode in range(ndims) if mode not in fixed_modes]

    # Optimal Rescaling (start of loop)

    # Compute factors
    for mode in modes_list:
        kraot = tl.tenalg.khatri_rao(factors, skip_matrix=mode).T
        factors[mode], cnt = mu.mu_betadivmin(factors[mode], kraot,
            tl.unfold(tensor,mode), beta, l2weight=l2weights[mode], l1weight=l1weights[mode], epsilon=epsilon, iter_inner=iter_inner,
            acc_delta=acc_delta)
        
        inner_cnt.append(cnt)

    # Optimal Rescaling (end of loop)
    if not disable_rebalance:
        # Step 1: put true zeroes in factors and core, retain mask in memory
        for i in range(ndims):
            factors[i][factors[i]<=epsilon]=0
        # 2. rebalance column by column
        for q in range(rank):  # TODO all at once?
            # Check if one factor is below threshold
            # in that case, scales will be nothing, all factors should be epsilon
            thresh = tl.prod(
                [tl.sum(tl.abs(factors[i][:, q])) for i in range(ndims)]
            )
            if thresh == 0:
                for submode in range(ndims):
                    factors[submode][:, q] = 0
            else:
                regs = [
                    l1weights[i] * tl.sum(tl.abs(factors[i][:, q]))
                    + l2weights[i] * tl.norm(factors[i][:, q]) ** 2
                    for i in range(ndims)
                ]
                scales = cp_opt_balance(tl.tensor(regs), hom_deg)
                for submode in range(ndims):
                    factors[submode][:, q] = (
                        factors[submode][:, q] * scales[submode]
                    )
        # Step 3: impute epsilon in place of values in [0, epsilon]
        for i in range(ndims):
            factors[i][factors[i]<=epsilon]=epsilon

    # Compute the value of the objective (loss) function
    cost_fcn = beta_div(tensor, tl.cp_tensor.cp_to_tensor((None,factors)), beta)
    regs_facs = [l1weights[i]*tl.sum(tl.abs(factors[i])) + l2weights[i]*tl.sum(factors[i]**2) for i in range(ndims)]
    cost_fcn += sum(regs_facs)

    return factors, cost_fcn, inner_cnt






#def one_sntd_step_mu_HER(tensor, l2weights=0, l1weights=0, core=0, factors=0, core_y=0, factors_y=0, beta=2,
                   #fixed_modes=[], alpha=0, epsilon=1e-12, alpha0=0, alphamax=0, alpha_increase=0, alpha_reduce=0, alphamax_increase=0, cost_fct_vals=0, iter_inner=50, acc_delta=0.5, opt_rescale=True):
    #"""
    #TODO: remove. we wont do HER for the paper 
    #"""
    
    #factors_up = factors.copy()
    #core_up = core.copy()
    #ndims = tl.ndim(tensor)

    ## Store the value of the objective (loss) function at the current
    ## iterate (factors_y, core_n).
    #cost0_fct_vals= cost_fct_vals[-1]

    ## Storing the inner iterations count
    #inner_cnt = []

    ## Generating the mode update sequence
    #modes_list = [mode for mode in range(ndims) if mode not in fixed_modes]

    ## Compute the extrapolated update for the factors.
    ## Note that when alpha is zero, factors_y = factors.
    #for mode in modes_list:
        #factors_up[mode], cnt = mu.mu_betadivmin(factors_y[mode], tl.unfold(tl.tenalg.multi_mode_dot(core_y, factors_y, skip = mode), mode),
            #tl.unfold(tensor,mode), beta, l2weight=l2weights[mode], l1weight=l1weights[mode], epsilon=epsilon, iter_inner=iter_inner,
            #acc_delta=acc_delta)
        
        ## Optimal Rescaling
        ## TODO: discuss intrication with extrapolation
        #if opt_rescale:
            #regs = [l1weights[i]*np.sum(np.abs(factors_up[i])) + l2weights[i]*np.linalg.norm(factors_up[i])**2 for i in range(ndims)]
            #regs += [l1weights[-1]*np.sum(np.abs(core_up)) + l2weights[-1]*tl.norm(core_up)**2] 
            #hom_deg = [1.0*(l1weights[i]>0) + 2.0*(l2weights[i]>0) for i in range(ndims+1)] # +1 for the core
            #scales = opt_scaling(np.array(regs),np.array(hom_deg))
            #for submode in range(ndims):
                #factors_up[submode] = factors_up[submode]*scales[submode]
                ##factors_y[submode] = factors_y[submode]*scales[submode]
                ## should also scale factorsY?
            #core_up = core_up*scales[-1]
            ##core_y = core_y*scales[-1]

        ## Extrapolation
        #factors_y[mode] = np.maximum(factors_up[mode]+alpha*(factors_up[mode]-factors[mode]),epsilon)


        #inner_cnt.append(cnt)
    ## Compute the extrapolated update for the core.
    ## Note that when alpha is zero, core_y = core_n.
    #core_up, cnt = mu.mu_tensorial(core_y, factors_y, tensor, beta, l2weight=l2weights[-1], l1weight=l1weights[-1],
                                 #epsilon=epsilon, iter_inner=iter_inner, acc_delta=acc_delta)

    ## Optimal Rescaling (core step)
    #if opt_rescale:
        #regs = [l1weights[i]*np.sum(np.abs(factors_up[i])) + l2weights[i]*np.linalg.norm(factors_up[i])**2 for i in range(ndims)]
        #regs += [l1weights[-1]*np.sum(np.abs(core)) + l2weights[-1]*tl.norm(core)**2] 
        #hom_deg = [1.0*(l1weights[i]>0) + 2.0*(l2weights[i]>0) for i in range(ndims+1)] # +1 for the core
        #scales = opt_scaling(np.array(regs),np.array(hom_deg))
        #for submode in range(ndims):
            #factors_up[submode] = factors_up[submode]*scales[submode]
        #core_up = core_up*scales[-1]

    #core_y = np.maximum(core_up+alpha*(core_up-core), epsilon)
    #inner_cnt.append(cnt)

    ## Compute the value of the objective (loss) function at the
    ## extrapolated solution for the factors (factors_y) and the
    ## non-extrapolated solution for the core (core_n).
    ## ---> No, we only did that for fast computation. Here there is no such fast comp, so we do it on the true estimates
    #cost_fcn = beta_div.beta_divergence(tensor, tl.tenalg.multi_mode_dot(core_up, factors_up), beta)+ l2weights[-1]*tl.norm(core_up)**2 + l1weights[-1]*tl.sum(core_up)
    #regs_facs = [l1weights[i]*np.sum(np.abs(factors_up[i])) + l2weights[i]*np.linalg.norm(factors_up[i])**2 for i in range(ndims)]
    #cost_fcn += np.sum(regs_facs)

    ## Update the extrapolation parameters following Algorithm 3 of
    ## Ang & Gillis (2019).
    #if(cost_fcn >=cost0_fct_vals):
        ## The solution did not improve, so restart the extrapolation scheme.
        ## Factors secondary sequence is set to the primary one
        ## output factors are the inputs, not the updated factors
        ## reducing alpha
        ## keeping same cost
        #factors_y = factors.copy()
        #core_y = core.copy()
        #alphamax = alpha0 
        #alpha = alpha_reduce*alpha
        #cost_fcn_out = cost0_fct_vals
    #else:
        ## The solution improved; retain the basic coordinate ascent
        ## update as well.
        ## Output updated factors, increase alpha, update cost for output
        #factors = factors_up
        #core = core_up
        #alpha = np.minimum(alphamax,alpha*alpha_increase)
        #alpha0 = alpha
        ##alphamax = np.minimum(0.99,alphamax_increase*alphamax)
        #alphamax = np.minimum(2,alphamax_increase*alphamax) # hard cap at 2
        #cost_fcn_out = cost_fcn

    #return core, factors, core_y, factors_y, cost_fcn_out, alpha, alpha0, alphamax, inner_cnt