#!/usr/bin/env python
# coding: utf-8


# Python classics
from modulefinder import STORE_NAME
import numpy as np
import tensorly as tl
import mu_ntd.algorithms.Sparse_ntd as SNTD
import matplotlib.pyplot as plt
import pandas as pd

# custom toolbox
#import shootout as sho
from shootout.methods.runners import run_and_track

# todo shootout: write report with parameters
nb_seeds = 1 # 0 for only plotting
name_store = "xp_ex_noex_01-09-22DUMMY"
variables={    
    "U_lines" : [40],#,200],
    "V_lines" : [40],
    "beta" : [1],
    "ranks" : [[4,5,6]], #todo split?
    "accelerate": [0.01],
    "iter_inner": [20],
    "extrapolate": [False, False],
    "l1weight": [0, 1e8, 1e6, 100], # works? syntax?
    "SNR" : [80],
    "sparse_data": [False] # do each test manually for nicer plots
        }
@run_and_track(algorithm_names="l1 MU", path_store="./Results/", name_store=name_store,
                verbose=True, nb_seeds=nb_seeds, seeded_fun=True, **variables)
def script_run(
    U_lines = 100,
    V_lines = 101,
    W_lines = 20,
    ranks = [4,5,6],
    SNR = 20,
    tol = 0,
    n_iter_max = 500,
    beta = 1,
    iter_inner = 3,
    #l2weight = [1, 1, 1, 0],  #(\mu_W, \mu_H, \mu_Q, \mu_g)
    #l1weight = [0, 0, 0, 1],  #(\mu_W, \mu_H, \mu_Q, \mu_g)
    l1weight = 0,  # \mu_g)
    l2weight = 1e-8,
    verbose=False,
    extrapolate = False,
    accelerate = False,
    seed = 0, # will be populated by `seed` in shootout
    sparse_data = False
    ):
    # weights
    l2weight = [l2weight, l2weight, l2weight, 0]
    l1weight = [0, 0, 0, l1weight]
    if l1weight==0:
        l2weight[-1] = l2weight[0] #setting weight on core if no sparsity
    # Seeding 
    rng = np.random.RandomState(seed+hash("sNTD"))
    # Generation of the input data tensor T # dense
    factors_0 = []
    if not sparse_data:
        factors_0.append(rng.rand(U_lines, ranks[0]))
        factors_0.append(rng.rand(V_lines, ranks[1]))
        factors_0.append(rng.rand(W_lines, ranks[2]))
    else:
        # sparse generation using truncated Gaussian
        W = rng.randn(U_lines, ranks[0])
        H = rng.randn(V_lines, ranks[1])
        Q = rng.randn(W_lines, ranks[2])
        W[W<0]=0
        H[H<0]=0
        Q[Q<0]=0
        factors_0.append(W)
        factors_0.append(H)
        factors_0.append(Q)
    core_0 = rng.rand(ranks[0], ranks[1], ranks[2])
    if sparse_data:
        # TODO: care for 0 core slices, regenerate if columns-rows-fibers are 0
        core_0 = rng.randn(ranks[0], ranks[1], ranks[2])
        core_0[core_0<0]=0 #sparsifying the gt solution

    # generate noise according to input SNR
    Ttrue = tl.tenalg.multi_mode_dot(core_0, factors_0) 
    N = rng.rand(U_lines, V_lines, W_lines) #1e-2
    sigma = 10**(-SNR/20)*np.linalg.norm(Ttrue)/np.linalg.norm(N) 
    T = Ttrue + sigma*N

    # Random initialization for the NTD
    factors_init = []
    factors_init.append(rng.rand(U_lines, ranks[0]))
    factors_init.append(rng.rand(V_lines, ranks[1]))
    factors_init.append(rng.rand(W_lines, ranks[2]))
    core_init = rng.rand(ranks[0], ranks[1], ranks[2])

    # Solver parameters
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Call of solvers
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # ### Beta = 1 - MU no acceleration, fixed 2 inner
    core, factors, cost_fct_vals, toc, alpha = SNTD.sntd_mu(T, ranks, l2weights=l2weight, l1weights=l1weight, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                          fixed_modes = [], normalize = 4*[None], verbose = verbose, return_costs = True, extrapolate=extrapolate, iter_inner=iter_inner, accelerate=accelerate)
    #print(alpha)
    print(core)

    #----------------------------------------------
    # Post-processing for checking identification
    #----------------------------------------------

    # normalisation
    #for i in range(len(factors)):
        #factors[i] = factors[i]/np.linalg.norm(factors[i],axis=0)
        #factors_HER[i] = factors_HER[i]/np.linalg.norm(factors_HER[i],axis=0)
        #factors_0[i] = factors_0[i]/np.linalg.norm(factors_0[i],axis=0)
    
    # Global sparsity tracking
    #nb_entries = U_lines*ranks[0] + V_lines*ranks[1] + W_lines*ranks[2] + np.prod(ranks)
    #nnz = np.sum(core>1e-8) + np.sum([np.sum(fac>1e-8) for fac in factors])
    #sparsity_ratio = nnz/nb_entries # 1 for dense, 0 for null
    
    # Core sparsity tracking
    nb_entries = np.prod(ranks)
    nnz = np.sum(core>1e-8) 
    sparsity_ratio = nnz/nb_entries # 1 for dense, 0 for null

    return {"errors": [cost_fct_vals], "timings": [toc], "sparsity": sparsity_ratio}#, "alpha":[alpha,alpha_HER], "congruence": [factors[2].T@factors_0[2],factors_HER[2].T@factors_0[2]]}

# Plotting
name_read = "Results/"+name_store
df = pd.read_pickle(name_read)

import plotly.express as px
import shootout.methods.post_processors as pp
import shootout.methods.plotters as pt

# small tweaks to variables to adjust ranks #TODO adjust variables
variables.pop("ranks")
# 0. Interpolating time (choose fewer points for better vis)
df = pp.interpolate_time_and_error(df, npoints = 100)
# 1. Convergence Plots
df_conv_it = pp.df_to_convergence_df(df,other_names=list(variables.keys()),groups=True,groups_names=list(variables.keys()))
df_conv_time = pp.df_to_convergence_df(df,other_names=list(variables.keys()),groups=True,groups_names=list(variables.keys()), err_name="errors_interp", time_name="timings_interp")
# renaming errors and time for convenience
df_conv_time = df_conv_time.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
# 2. Converting to median for iterations and timings
df_conv_median_it = pp.median_convergence_plot(df_conv_it, type="iterations")
df_conv_median_time = pp.median_convergence_plot(df_conv_time, type="timings")
# downsampling it by 5 for plotting (100 points per plot)
df_conv_median_it = df_conv_median_it[df_conv_median_it["it"]%5==0]
# 3. Making plots
fig = px.line(df_conv_median_it, x="it", y="errors", color="extrapolate", log_y=True, facet_col="U_lines", facet_row="l1weight", error_y="q_errors_p", error_y_minus="q_errors_m", template="plotly_white") #sparsity? zip with extrapolate? will fail because groups are removed?
fig2 = px.line(df_conv_median_time, x="timings", y="errors", color="extrapolate", log_y=True, facet_col="U_lines", facet_row="l1weight", error_y="q_errors_p", error_y_minus="q_errors_m", template="plotly_white")
# smaller linewidth
fig.update_traces(
    selector=dict(),
    line_width=3,
    error_y_thickness = 0.3
)
fig2.update_traces(
    selector=dict(),
    line_width=3,
    error_y_thickness = 0.3
)
# time
#fig2 = px.line(df_conv, x="timings", y="errors", color="accelerate", log_y=True, line_group="groups", facet_col="iter_inner", facet_row="extrapolate")

# showing all
#fig.show()
#fig2.show()