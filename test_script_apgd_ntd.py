#!/usr/bin/env python
# coding: utf-8

# Python classics
import numpy as np
import tensorly as tl
#import mu_ntd.algorithms.ntd as NTD
# import nnfac instead, beta=2 MU NTD and HALS are implemented (but not with l1 + l2...)
# I will mod the relevant code in nnfac if we need comparison with beta=2
# Some tweaks are needed with the error computation, and add extrapolation
# run pip install -e . locally in nn_fac
from nn_fac import ntd as NTD
import mu_ntd.algorithms.ntd as APGD
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Running...")
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Data generation
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    U_lines = 25
    V_lines = 20
    W_lines = 10
    ranks = [8,12,10]
    # Generation of the input data tensor T
    factors_0 = []
    factors_0.append(np.random.rand(U_lines, ranks[0]))
    factors_0.append(np.random.rand(V_lines, ranks[1]))
    factors_0.append(np.random.rand(W_lines, ranks[2]))
    core_0 = np.random.rand(ranks[0], ranks[1], ranks[2])
    T = tl.tenalg.multi_mode_dot(core_0, factors_0) + 1e-2 * np.random.rand(U_lines, V_lines, W_lines)
  
    # Random initialization for the NTD
    factors_0 = []
    factors_0.append(np.random.rand(U_lines, ranks[0]))
    factors_0.append(np.random.rand(V_lines, ranks[1]))
    factors_0.append(np.random.rand(W_lines, ranks[2]))
    core_0 = np.random.rand(ranks[0], ranks[1], ranks[2])
    
    # Solver parameters
    n_iter_max = 200
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Call of solvers
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # ### APGD No HER
    core_NoHER_APGD, factors_NoHER_APGD, cost_fct_vals_NoHER_APGD, toc_NoHER_APGD, alpha_NoHerAPGD = APGD.ntd_apgd(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = n_iter_max, beta = 2, sparsity_coefficients = None, fixed_modes = [], normalize = None, verbose = True, return_costs = True, extrapolate=False)
    
    # ### APGD HER
    core_HER_APGD, factors_HER_APGD, cost_fct_vals_HER_APGD, toc_HER_APGD, alpha_APGD = APGD.ntd_apgd(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = n_iter_max, beta = 2, sparsity_coefficients = None, fixed_modes = [], normalize = None, verbose = True, return_costs = True, extrapolate=True)

   # ------------------ Axel's codes -------------- #
   # Why not import nnfac or tensorly?

    # ### Beta = 2 - MU no extrapolation as in nn_fac
    core, factors, cost_fct_vals, toc = NTD.ntd_mu(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = n_iter_max, tol = 1e-6, beta = 2,
                                                sparsity_coefficients = None, fixed_modes = [], normalize = None, verbose = True, return_costs = True)
    # ### HALS as in nn_fac
    core_HALS, factors_HALS, cost_fct_vals_HALS, toc_HALS = NTD.ntd(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = n_iter_max, tol = 1e-6,
                                                sparsity_coefficients = None, fixed_modes = [], normalize = None, verbose = True, return_costs = True)
    

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Reporting
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-----------------------------------------------------------------------")
    print("Reporting results:")
    print("-----------------------------------------------------------------------")
    print("Final loss function value:")
    print(f"APGD-NOHER   : {cost_fct_vals_NoHER_APGD[-1]}, converged in {len(cost_fct_vals_NoHER_APGD) - 1} iterations.")
    print(f"APGD         : {cost_fct_vals_HER_APGD[-1]}, converged in {len(cost_fct_vals_HER_APGD) - 1} iterations.")
    print(f"MU, Beta = 2 : {cost_fct_vals[-1]}, converged in {len(cost_fct_vals) - 1} iterations.")
    print(f"HALS:        : {cost_fct_vals_HALS[-1]}, converged in {len(cost_fct_vals_HALS) - 1} iterations.")
    print("-----------------------------------------------------------------------")
    print("Final relative loss function value:")
    print(f"APGD-NO HER  : {cost_fct_vals_NoHER_APGD[-1]/tl.norm(T)**2*100} %")
    print(f"APGD         : {cost_fct_vals_HER_APGD[-1]/tl.norm(T)**2*100} %")
    print(f"MU, Beta = 2 : {cost_fct_vals[-1]/tl.norm(T)**2*100} %")
    print(f"HALS         : {cost_fct_vals_HALS[-1]/tl.norm(T)**2*100} %")
    
    plt.figure(1)
    plt.plot(cost_fct_vals_HER_APGD[5:-1], color='black', label='APGD HER on')
    plt.plot(cost_fct_vals_NoHER_APGD[5:-1], color='red', label='APGD HER off')
    plt.plot(cost_fct_vals[5:-1], color='blue', label='MU')
    plt.plot(cost_fct_vals_HALS[5:-1], color='orange', label='HALS')
    plt.xlabel('Iteration number')
    plt.ylabel('Objective function')
    plt.title('Frobenius NTD')
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.plot(toc_NoHER_APGD[4:-1],cost_fct_vals_HER_APGD[5:-1], color='black', label='APGD HER on')
    plt.plot(toc_HER_APGD[4:-1],cost_fct_vals_NoHER_APGD[5:-1], color='red', label='APGD HER off')
    plt.plot(toc[4:-1],cost_fct_vals[4:-1], color='blue', label='MU')
    plt.plot(toc_HALS[4:-1],cost_fct_vals_HALS[4:-1], color='orange', label='HALS')
    plt.xlabel('CPU time')
    plt.ylabel('Objective function')
    plt.title('Frobenius NTD')
    plt.legend()
    plt.show()