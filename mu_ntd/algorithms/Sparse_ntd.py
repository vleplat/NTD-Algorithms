# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 16:52:21 2019

@author: veplat, based on amarmoret code
"""

import numpy as np
import time
import tensorly as tl
from tensorly.decomposition import tucker as tl_tucker

import mu_ntd.algorithms.Sparse_mu_epsilon as mu
import nn_fac.errors as err
import nn_fac.beta_divergence as beta_div

import numpy as np


######################### Temporary, to test mu and not break everything
def sntd_mu(tensor, ranks, muWeight, init = "random", core_0 = None, factors_0 = [], n_iter_max=100, tol=1e-6,
           sparsity_coefficients = [], fixed_modes = [], normalize = [], mode_core_norm = None, beta = 2,
           verbose=False, return_costs=False, deterministic=False, extrapolate=False):
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

    if extrapolate:
        return compute_sntd_mu_HER(tensor, ranks, muWeight, core, factors, n_iter_max=n_iter_max,
                       sparsity_coefficients = sparsity_coefficients, fixed_modes = fixed_modes,
                       normalize = normalize, mode_core_norm = mode_core_norm,
                       verbose=verbose, return_costs=return_costs, beta = beta, deterministic = deterministic)
    else:
        return compute_sntd_mu(tensor, ranks, muWeight, core, factors, n_iter_max=n_iter_max, tol=tol,
                       sparsity_coefficients = sparsity_coefficients, fixed_modes = fixed_modes,
                       normalize = normalize, mode_core_norm = mode_core_norm,
                       verbose=verbose, return_costs=return_costs, beta = beta, deterministic = deterministic)


def compute_sntd_mu_HER(tensor_in, ranks, muWeight, core_in, factors_in, n_iter_max=100,
           sparsity_coefficients = [], fixed_modes = [], normalize = [], beta = 2, mode_core_norm=None,
           verbose=False, return_costs=False, deterministic=False):

    # initialisation - store the input varaibles
    core = core_in.copy()
    factors = factors_in.copy()
    tensor = tensor_in

    norm_tensor = tl.norm(tensor, 2)

    # set init if problem
    #TODO: set them as warnings
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
    epsilon = 1e-12

    # the extrapolation parameters
    alpha=0.05
    print('Initial Alpha={}'.format(alpha))

    alpha0 = alpha             #extrapolation parameter setting from last improvement
    alphamax = 1               #1 but Andy told us to increase it to have fun, let us see
    alpha_increase = 1.1       #1.1 max
    alpha_reduce = 0.8         #0.75
    alphamax_increase  = 1.05

    core_n = core.copy()        # non-extrapolated factor estimates
    factors_n = factors.copy()  # non-extrapolated core estimates
    core_y = core.copy()        # extrapolated factor estimates
    factors_y = factors.copy()  # extrapolated core estimates

    # Iterate over one step of NTD
    for iteration in range(n_iter_max):

        # One pass of MU on each updated mode
        core, factors, core_n, factors_n, core_y, factors_y, cost, cost_fycn, alpha, alpha0, alphamax = one_sntd_step_mu_HER(tensor, ranks, muWeight, core, factors, core_n, factors_n, core_y, factors_y, beta, norm_tensor,
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

        # if iteration > 0 and (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2])) < tol:
        #     # Stop condition: relative error between last two iterations < tol
        #     if verbose:
        #         print('Converged to the required tolerance in {} iterations.'.format(iteration))
        #     break

    if return_costs:
        return core, factors, cost_fct_vals, toc, alpha_store
    else:
        return core, factors

def one_sntd_step_mu_HER(tensor, ranks, muWeight, in_core, in_factors, in_core_n, in_factors_n, in_core_y, in_factors_y, beta, norm_tensor,
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

    # Compute the extrapolated update for the factors.
    # Note that when alpha is zero, factors_y = factors_n.
    for mode in modes_list:
        factors_n_up[mode] = mu.mu_betadivmin(factors_y[mode], tl.unfold(tl.tenalg.multi_mode_dot(core_y, factors_y, skip = mode), mode), tl.unfold(tensor,mode), beta, muWeight[mode+1])
        factors_y[mode] = np.maximum(factors_n_up[mode]+alpha*(factors_n_up[mode]-in_factors_n[mode]),epsilon)
    # Compute the extrapolated update for the core.
    # Note that when alpha is zero, core_y = core_n.
    core_n_up = mu.mu_tensorial(core_y, factors_y, tensor, beta, muWeight[0])
    core_y = np.maximum(core_n+alpha*(core_n-in_core_n),epsilon)

    # Compute the value of the objective (loss) function at the
    # extrapolated solution for the factors (factors_y) and the
    # non-extrapolated solution for the core (core_n).
    cost_fycn = beta_div.beta_divergence(tensor, tl.tenalg.multi_mode_dot(core_n, factors_y), beta)+muWeight[0]*tl.norm(core_n, order=1)
    for mode in modes_list:
        cost_fycn = cost_fycn + 1/2*muWeight[mode+1]*np.linalg.norm(factors_y[mode],'fro')**2

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
        alphamax = np.minimum(0.99,alphamax_increase*alphamax)

    # If the solution improves the "current best" estimate, update the
    # current best estimate using the non-extrapolated estimates of the
    # core (core_n) and the extrapolated estimates of the factors (factors_y).
    if(cost_fycn < cost_fct_vals[-1]):
        factors = factors_y.copy()
        core = core_n_up.copy()
        cost_fct_val = cost_fycn


    return core, factors, core_n, factors_n, core_y, factors_y, cost_fct_val, cost_fycn, alpha, alpha0, alphamax

def compute_sntd_mu(tensor_in, ranks, muWeight, core_in, factors_in, n_iter_max=100, tol=1e-6,
           sparsity_coefficients = [], fixed_modes = [], normalize = [], beta = 2, mode_core_norm=None,
           verbose=False, return_costs=False, deterministic=False):

    # initialisation - store the input varaibles
    core = core_in.copy()
    factors = factors_in.copy()
    tensor = tensor_in

    norm_tensor = tl.norm(tensor, 2)

    # set init if problem
    #TODO: set them as warnings
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
    cost_fct_vals = []
    tic = time.time()
    toc = []
    epsilon = 1e-12
    # initialisation - unfold the tensor according to the modes
    #unfolded_tensors = []
    #for mode in range(tl.ndim(tensor)):
    #   unfolded_tensors.append(tl.base.unfold(tensor, mode))

    # Iterate over one step of NTD
    for iteration in range(n_iter_max):
        # One pass of least squares on each updated mode
        core, factors, cost = one_sntd_step_mu(tensor, ranks, muWeight, core, factors, beta, norm_tensor,
                                              fixed_modes, normalize, mode_core_norm,epsilon)

        # Store the computation time
        toc.append(time.time() - tic)

        cost_fct_vals.append(cost)

        if verbose:
            if iteration == 0:
                print('Initial Obj={}'.format(cost))
            else:
                if cost_fct_vals[-2] - cost_fct_vals[-1] > 0:
                    print('Iter={}|Obj={}| Var={} (target is {}).'.format(iteration,
                            cost_fct_vals[-1], (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2])),tol))
                else:
                    # print in red when the reconstruction error is negative (shouldn't happen)
                    print('\033[91m' + 'Iter={}|Obj={}| Var={} (target is {}).'.format(iteration,
                            cost_fct_vals[-1], (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2])),tol) + '\033[0m')

        if iteration > 0 and (abs(cost_fct_vals[-2] - cost_fct_vals[-1])/abs(cost_fct_vals[-2])) < tol:
            # Stop condition: relative error between last two iterations < tol
            if verbose:
                print('Converged to the required tolerance in {} iterations.'.format(iteration))
            break

    if return_costs:
        return core, factors, cost_fct_vals, toc
    else:
        return core, factors

def one_sntd_step_mu(tensor, ranks, muWeight, in_core, in_factors, beta, norm_tensor,
                   fixed_modes, normalize, mode_core_norm, epsilon):
    # Copy
    core = in_core.copy()
    factors = in_factors.copy()

    # Generating the mode update sequence
    modes_list = [mode for mode in range(tl.ndim(tensor)) if mode not in fixed_modes]

    for mode in modes_list:
        factors[mode] = mu.mu_betadivmin(factors[mode], tl.unfold(tl.tenalg.multi_mode_dot(core, factors, skip = mode), mode), tl.unfold(tensor,mode), beta, muWeight[mode+1])

    core = mu.mu_tensorial(core, factors, tensor, beta, muWeight[0])

    if normalize[-1]:
        unfolded_core = tl.unfold(core, mode_core_norm)
        for idx_mat in range(unfolded_core.shape[0]):
            if tl.norm(unfolded_core[idx_mat]) != 0:
                unfolded_core[idx_mat] = unfolded_core[idx_mat] / tl.norm(unfolded_core[idx_mat], 2)
        core = tl.fold(unfolded_core, mode_core_norm, core.shape)


    reconstructed_tensor = tl.tenalg.multi_mode_dot(core, factors)

    cost_fct_val = beta_div.beta_divergence(tensor, reconstructed_tensor, beta)+muWeight[0]*tl.norm(core, order=1)
    for mode in modes_list:
        cost_fct_val = cost_fct_val+1/2*muWeight[mode+1]*np.linalg.norm(factors[mode],'fro')**2

    return core, factors, cost_fct_val #  exhaustive_rec_error








