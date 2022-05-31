#!/usr/bin/env python
# coding: utf-8


# TODO:
# - Arbitrary number of inner iterations + heuristic Gillis-like
# - Sparse l1 reg in MU
# - Industry-scale tests

# Python classics
import numpy as np
import tensorly as tl
import mu_ntd.algorithms.Sparse_ntd as SNTD
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Running...")
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Data generation
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    U_lines = 250
    V_lines = 200
    W_lines = 100
    ranks = [4,6,5]
    # Noise level
    sigma = 1e-2
    # Generation of the input data tensor T
    factors_0 = []
    factors_0.append(np.random.rand(U_lines, ranks[0]))
    factors_0.append(np.random.rand(V_lines, ranks[1]))
    factors_0.append(np.random.rand(W_lines, ranks[2]))
    core_0 = np.random.rand(ranks[0], ranks[1], ranks[2])
    factors_GT = factors_0
    core_GT = core_0
    T = tl.tenalg.multi_mode_dot(core_0, factors_0) + sigma * np.random.rand(U_lines, V_lines, W_lines) #1e-2
  
    # Random initialization for the NTD
    factors_0 = []
    factors_0.append(np.random.rand(U_lines, ranks[0]))
    factors_0.append(np.random.rand(V_lines, ranks[1]))
    factors_0.append(np.random.rand(W_lines, ranks[2]))
    core_0 = np.random.rand(ranks[0], ranks[1], ranks[2])
    
    # Solver parameters
    n_iter_max = 200
    beta = 1
    muWeight = np.array([0.2, 0.15, 0.15, 0.15])  #(\mu_g, \mu_W, \mu_H, \mu_Q)
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Call of solvers
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    
    # ### Beta = 1 - MU no extrapolation
    core, factors, cost_fct_vals, toc = SNTD.sntd_mu(T, ranks, muWeight, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = n_iter_max, tol = 1e-6, beta = beta,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=False)
    # ### Beta = 1 - MU extrapolation
    core_HER, factors_HER, cost_fct_vals_HER, toc_HER, alpha = SNTD.sntd_mu(T, ranks, muWeight, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = n_iter_max, beta = beta,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=True)
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Reporting
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    print("-----------------------------------------------------------------------")
    print("Reporting results:")
    print("-----------------------------------------------------------------------")
    print("Final loss function value:")
    print(f"MU, Beta = 1       : {cost_fct_vals[-1]}, converged in {len(cost_fct_vals) - 1} iterations.")
    print(f"MU - HER, Beta = 1 : {cost_fct_vals_HER[-1]}, converged in {len(cost_fct_vals_HER) - 1} iterations.")
    print("-----------------------------------------------------------------------")
    print("Final relative construction error:")
    print(f"MU, Beta = 1       : {tl.norm(T-tl.tenalg.multi_mode_dot(core, factors))/tl.norm(T)*100} %")
    print(f"MU - HER, Beta = 1 : {tl.norm(T-tl.tenalg.multi_mode_dot(core_HER, factors_HER))/tl.norm(T)*100} %")
    
    # first iteration shown?
    it1 = 0

    plt.figure(1)
    plt.semilogy(cost_fct_vals[it1:], color='blue', label='MU HER off')
    plt.semilogy(cost_fct_vals_HER[it1:], color='black', label='MU HER on')
    plt.xlabel('Iteration number')
    plt.ylabel('Objective function')
    plt.title('Sparse beta-div NTD')
    plt.legend()
    plt.show()
    
    plt.figure(2)
    plt.semilogy(toc[it1:],cost_fct_vals[it1:], color='blue', label='MU HER off')
    plt.semilogy(toc_HER[it1:],cost_fct_vals_HER[it1:], color='black', label='MU HER on')
    plt.xlabel('CPU time')
    plt.ylabel('Objective function')
    plt.title('Sparse beta-div NTD')
    plt.legend()
    plt.show()
    