#!/usr/bin/env python
# coding: utf-8


# TODO:
# - heuristic Gillis-like for stopping inner iters ?

# Python classics
import numpy as np
import tensorly as tl
import mu_ntd.algorithms.Sparse_ntd as SNTD
import matplotlib.pyplot as plt
import pandas as pd

# custom toolbox
#import shootout as sho
from shootout.methods.runners import run_and_track

# todo shootout: write report with parameters
variables={    
    "U_lines" : 20,
    "V_lines" : 20,
    "ranks" : [[4,5,6]],
    "sigma" : 0,
    "iter_inner" : 10
        }
#U_lines = [20,100],
#V_lines = [20,100],
#ranks=[[4,5,6],[10,2,2],[15,15,15]],
#sigma=[0,1e-2],
#iter_inner= [1,3,10])
@run_and_track(algorithm_names=["l1l2 MU", "l1l2 MU with HER"], path_store="./Results/",
                verbose=True, nb_seeds=2,**variables)
def script_run(
    U_lines = 100,
    V_lines = 101,
    W_lines = 20,
    ranks = [4,5,6],
    sigma = 1e-2,
    tol = 0,
    n_iter_max = 500,
    beta = 1,
    iter_inner = 3,
    l2weight = [1, 0, 1, 0],  #(\mu_W, \mu_H, \mu_Q, \mu_g)
    l1weight = [0, 1, 0, 1],  #(\mu_W, \mu_H, \mu_Q, \mu_g)
    verbose=False
    ):
     #running all iterations
    # Generation of the input data tensor T
    factors_0 = []
    #factors_0.append(np.random.rand(U_lines, ranks[0]))
    #factors_0.append(np.random.rand(V_lines, ranks[1]))
    #factors_0.append(np.random.rand(W_lines, ranks[2]))
    # sparse generation
    W = np.random.randn(U_lines, ranks[0])
    H = np.random.randn(V_lines, ranks[1])
    Q = np.random.randn(W_lines, ranks[2])
    W[W<0]=0
    H[H<0]=0
    Q[Q<0]=0
    factors_0.append(W)
    factors_0.append(H)
    factors_0.append(Q)
    core_0 = np.random.randn(ranks[0], ranks[1], ranks[2])
    core_0[core_0<0]=0 #sparsifying the gt solution
    factors_GT = factors_0
    core_GT = core_0
    T = tl.tenalg.multi_mode_dot(core_0, factors_0) + sigma * np.random.rand(U_lines, V_lines, W_lines) #1e-2

    # Random initialization for the NTD
    factors_init = []
    factors_init.append(np.random.rand(U_lines, ranks[0]))
    factors_init.append(np.random.rand(V_lines, ranks[1]))
    factors_init.append(np.random.rand(W_lines, ranks[2]))
    core_init = np.random.rand(ranks[0], ranks[1], ranks[2])

    # Solver parameters
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Call of solvers
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # ### Beta = 1 - MU no extrapolation no acceleration
    core, factors, cost_fct_vals, toc, alpha = SNTD.sntd_mu(T, ranks, l2weights=l2weight, l1weights=l1weight, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                                fixed_modes = [], normalize = 4*[None], verbose = False, return_costs = True, extrapolate=False, iter_inner=iter_inner, accelerate=False)
    # ### Beta = 1 - MU extrapolation and acceleration
    core_HER, factors_HER, cost_fct_vals_HER, toc_HER, alpha_HER = SNTD.sntd_mu(T, ranks, l2weights=l2weight, l1weights=l1weight, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                                fixed_modes = [], normalize = 4*[None], verbose = False, return_costs = True, extrapolate=True, iter_inner=iter_inner, accelerate=True)

    #----------------------------------------------
    # Post-processing for checking identification
    #----------------------------------------------

    # normalisation
    for i in range(len(factors)):
        factors[i] = factors[i]/np.linalg.norm(factors[i],axis=0)
        factors_HER[i] = factors_HER[i]/np.linalg.norm(factors_HER[i],axis=0)
        factors_0[i] = factors_0[i]/np.linalg.norm(factors_0[i],axis=0)

    return {"errors": [cost_fct_vals, cost_fct_vals_HER], "timings": [toc,toc_HER]}#, "alpha":[alpha,alpha_HER], "congruence": [factors[2].T@factors_0[2],factors_HER[2].T@factors_0[2]]}



#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Reporting
#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#print("-----------------------------------------------------------------------")
#print("Reporting results:")
#print("-----------------------------------------------------------------------")
#print("Final loss function value:")
#print(f"MU, Beta = 1       : {cost_fct_vals[-1]}, converged in {len(cost_fct_vals) - 1} iterations.")
#print(f"MU - HER, Beta = 1 : {cost_fct_vals_HER[-1]}, converged in {len(cost_fct_vals_HER) - 1} iterations.")
#print("-----------------------------------------------------------------------")
#print("Final relative construction error:")
#print(f"MU, Beta = 1       : {tl.norm(T-tl.tenalg.multi_mode_dot(core, factors))/tl.norm(T)*100} %")
#print(f"MU - HER, Beta = 1 : {tl.norm(T-tl.tenalg.multi_mode_dot(core_HER, factors_HER))/tl.norm(T)*100} %")

## first iteration shown?
#it1 = 0

#plt.figure(1)
#plt.semilogy(cost_fct_vals[it1:], color='blue', label='MU HER off')
#plt.semilogy(cost_fct_vals_HER[it1:], color='black', label='MU HER on')
#plt.xlabel('Iteration number')
#plt.ylabel('Objective function')
#plt.title('Sparse beta-div NTD')
#plt.legend()
#plt.show()

#plt.figure(2)
#plt.semilogy(toc[it1:],cost_fct_vals[it1:], color='blue', label='MU HER off')
#plt.semilogy(toc_HER[it1:],cost_fct_vals_HER[it1:], color='black', label='MU HER on')
#plt.xlabel('CPU time')
#plt.ylabel('Objective function')
#plt.title('Sparse beta-div NTD')
#plt.legend()
#plt.show()
