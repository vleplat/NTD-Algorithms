#!/usr/bin/env python
# coding: utf-8

# Python classics
import numpy as np
import tensorly as tl
import mu_ntd.algorithms.ntd as NTD
import matplotlib.pyplot as plt

if __name__ == "__main__":
    print("Running...")
    U_lines = 100
    V_lines = 50
    W_lines = 10

    ranks = [4,6,5]

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

    # ## MU and PGD
    # ### Beta = 2 - MU no extrapolation
    core, factors, cost_fct_vals, toc = NTD.ntd_mu(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = 1000, tol = 1e-4, beta = 2,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=False)

    print(f"MU, Beta = 2: Final reconstruction error: {cost_fct_vals[-1]}, converged in {len(cost_fct_vals) - 1} iterations.")
    # ### Beta = 2 - MU extrapolation

    core_HER, factors_HER, cost_fct_vals_HER, toc_HER, alpha = NTD.ntd_mu(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = 1000, beta = 2,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=True)
    # ### Beta = 2 - PGD HER
    core_HER_PGD, factors_HER_PGD, cost_fct_vals_HER_PGD, toc_HER_PGD, alpha_PGD = NTD.ntd_PGD(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = 1000, beta = 2,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=True)

    print(f"MU, Beta = 2: Final reconstruction error: {cost_fct_vals_HER[-1]}, converged in {len(cost_fct_vals_HER) - 1} iterations.")
    plt.figure(1)
    plt.plot(cost_fct_vals_HER_PGD[2:-1], color='black', label='PGD HER on')
    plt.plot(cost_fct_vals_HER[2:-1], color='red', label='MU HER on')
    plt.plot(cost_fct_vals, color='blue', label='MU HER off')
    plt.xlabel('Iteration number')
    plt.ylabel('Objective function')
    plt.title('Beta-Div NTD: beta=%i' %2)
    plt.legend()
    plt.show()

    # ### Beta = 1 - MU no extrapolation
    core, factors, cost_fct_vals, toc = NTD.ntd_mu(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = 1000, tol = 1e-5, beta = 1,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=False)

    # ### Beta = 1 - MU extrapolation
    core_HER, factors_HER, cost_fct_vals_HER, toc_HER, alpha = NTD.ntd_mu(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = 1000, beta = 1,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=True)

    # ### Beta = 1 - PGD HER
    core_HER_PGD, factors_HER_PGD, cost_fct_vals_HER_PGD, toc_HER_PGD, alpha_PGD = NTD.ntd_PGD(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = 1000, beta = 1,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=True)

    print(f"MU, Beta = 1: Final reconstruction error: {cost_fct_vals_HER[-1]}, converged in {len(cost_fct_vals_HER) - 1} iterations.")
    plt.figure(2)
    plt.plot(cost_fct_vals_HER_PGD[2:-1], color='black', label='PGD HER on')
    plt.plot(cost_fct_vals_HER[2:-1], color='red', label='MU HER on')
    plt.plot(cost_fct_vals, color='blue', label='MU HER off')
    plt.xlabel('Iteration number')
    plt.ylabel('Objective function')
    plt.title('Beta-Div NTD: beta=%i' %1)
    plt.legend()
    plt.show()

    # ### Beta = 0 - MU no extrapolation
    core, factors, cost_fct_vals, toc = NTD.ntd_mu(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = 1000, tol = 1e-4, beta = 0,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=False)

    # ### Beta = 0 - MU extrapolation

    core_HER, factors_HER, cost_fct_vals_HER, toc_HER, alpha = NTD.ntd_mu(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = 1000, beta = 0,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=True)
    # ### Beta = 0 - PGD HER
    core_HER_PGD, factors_HER_PGD, cost_fct_vals_HER_PGD, toc_HER_PGD, alpha_PGD = NTD.ntd_PGD(T, ranks, init = "custom", core_0 = core_0, factors_0 = factors_0, n_iter_max = 1000, beta = 0,
                                                sparsity_coefficients = [None, None, None, None], fixed_modes = [], normalize = [False, False, False, False],
                                                verbose = True, return_costs = True, extrapolate=True)

    print(f"MU, Beta = 0: Final reconstruction error: {cost_fct_vals_HER[-1]}, converged in {len(cost_fct_vals_HER) - 1} iterations.")
    plt.figure(3)
    plt.plot(cost_fct_vals_HER_PGD[5:-1], color='black', label='PGD HER on')
    plt.plot(cost_fct_vals_HER[5:-1], color='red', label='MU HER on')
    plt.plot(cost_fct_vals[5:-1], color='blue', label='MU HER off')
    plt.xlabel('Iteration number')
    plt.ylabel('Objective function')
    plt.title('Beta-Div NTD: beta=%i' %0)
    plt.legend()
    plt.show()
