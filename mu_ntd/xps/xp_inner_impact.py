#!/usr/bin/env python
# coding: utf-8


# Python classics
from modulefinder import STORE_NAME
from sqlite3 import adapt
import numpy as np
import tensorly as tl
import mu_ntd.algorithms.Sparse_ntd as SNTD
import matplotlib.pyplot as plt
import pandas as pd

# custom toolbox
#import shootout as sho
from shootout.methods.runners import run_and_track

# todo shootout: write report with parameters
nb_seeds = 5 # 0 for only plotting
name_store = "xp_inner_impact_13-02-23"
variables={    
    "U_lines" : [20],#todo fatten for paper tests
    "V_lines" : [20],
    "beta" : [1],
    "ranks" : [[4,5,6]], #todo split?
    "accelerate": [False, 3e-1, 5e-1, 7e-1],# [False, 1e-1, 2e-1, 5e-1, 7e-1], no impact of 0.1 and 0.2
    "iter_inner": [50], #[1, 5, 10, 20, 50, 100, 200, 500],
    "extrapolate": [False],#, True],
    "SNR" : [80],
    "sparse_factors": [False], # should I check?
    "sparse_core": [True], # should I check?
    "scale": True,
    "l1weight": [0.5],
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
    n_iter_max = 1000,
    beta = 1,
    iter_inner = 3,
    #l2weight = 0,#[0, 0, 0, 0],  #(\mu_W, \mu_H, \mu_Q, \mu_g)
    #l1weight = 10,#[10, 10, 10, 10],  #(\mu_W, \mu_H, \mu_Q, \mu_g)
    l1weight = 0,
    l2weight = 0,
    verbose=False,
    extrapolate = False,
    accelerate = False,
    seed = 0, # will be populated by `seed` in shootout
    sparse_factors = False,
    sparse_core = False,
    scale = True
    ):
    # Adaptive max iter, linear scale
    n_iter_max = int(n_iter_max*100/np.maximum(iter_inner, 2)) # will be 1500 for 100 iter inner, 6000 when accelerate is 0.5 
    #if accelerate:
    #    n_iter_max = int(n_iter_max*(1+3*accelerate))

    # weights processings
    #l2weights = [l2weight, l2weight, l2weight, 0]
    l2weights = [l1weight, l1weight, l1weight, 0]
    l1weights = [0, 0, 0, l1weight]
    if l1weights==0:
        l2weights[-1] = l2weights[0] #setting ridge on core if no sparsity but ridge on factors
    # Seeding 
    rng = np.random.RandomState(seed+hash("sNTD")%(2**32))
    # Generation of the input data tensor T
    factors_0 = []
    if not sparse_factors:
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
    if sparse_core:
        core_0 = rng.rand(ranks[0], ranks[1], ranks[2])
        core_0[core_0<(0.5*np.median(core_0))]=0 #sparsifying the gt solution

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
    core, factors, cost_fct_vals, toc, _, cnt, _ = SNTD.sntd_mu(T, ranks, l2weights=l2weights, l1weights=l1weights, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                          fixed_modes = [], verbose = verbose, return_costs = True, extrapolate=extrapolate, iter_inner=iter_inner, accelerate=accelerate, opt_rescale=scale)

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
    nnz = np.sum(core>1e-12) 
    sparsity_ratio = nnz/nb_entries # 1 for dense, 0 for null

    return {"errors": [cost_fct_vals], "timings": [toc], "sparsity": sparsity_ratio, "inner_cnt": [cnt]}#, "alpha":[alpha,alpha_HER], "congruence": [factors[2].T@factors_0[2],factors_HER[2].T@factors_0[2]]}

# Plotting
name_read = "Results/"+name_store
df = pd.read_pickle(name_read)

import plotly.express as px
import shootout.methods.post_processors as pp
import shootout.methods.plotters as pt

# small tweaks to variables to adjust ranks #TODO adjust variables
variables.pop("ranks")
ovars = list(variables.keys())
# 0. Interpolating time (choose fewer points for better vis)
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True) #hope this works
# 1. Convergence Plots
df_conv_it = pp.df_to_convergence_df(df,other_names=ovars,groups=True,groups_names=ovars, max_time=np.Inf)
df_conv_time = pp.df_to_convergence_df(df,other_names=ovars,groups=True,groups_names=ovars, err_name="errors_interp", time_name="timings_interp", max_time=np.Inf)
# renaming errors and time for convenience
df_conv_time = df_conv_time.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
# 2. Converting to median for iterations and timings
df_conv_median_it = pp.median_convergence_plot(df_conv_it, type="iterations")
df_conv_median_time = pp.median_convergence_plot(df_conv_time, type="timings")
# downsampling it by 5 for plotting (100 points per plot)
df_conv_median_it = df_conv_median_it[df_conv_median_it["it"]%15==0] # manual tweak
# 3. Making plots
fig = px.line(df_conv_median_it, x="it", y="errors", color="accelerate", log_y=True, facet_row="iter_inner", error_y="q_errors_p", error_y_minus="q_errors_m", template="plotly_white")
fig2 = px.line(df_conv_median_time, x="timings", y="errors", color="accelerate", log_y=True, facet_row="iter_inner", error_y="q_errors_p", error_y_minus="q_errors_m", template="plotly_white")
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
fig.update_xaxes(matches=None)
#fig.update_yaxes(matches=None)
fig.update_xaxes(showticklabels=True)
#fig.update_yaxes(showticklabels=True)
#fig2.update_xaxes(matches=None)
#fig2.update_yaxes(matches=None)
#fig2.update_xaxes(showticklabels=True)
#fig2.update_yaxes(showticklabels=True)
# Figure showing cnt for each algorithm
# 1. make long format for cnt
# TODO: improve shootout to better handle this case
df_conv_cnt = pp.df_to_convergence_df(df, groups=True, groups_names=list(variables.keys()), err_name="inner_cnt", other_names=list(variables.keys()), time_name=False, max_time=False)
# 2. median plots
df_conv_median_cnt = pp.median_convergence_plot(df_conv_cnt, type=False, err_name="inner_cnt")

fig3 = px.line(df_conv_median_cnt, 
            x="it", 
            y= "inner_cnt", 
            color='accelerate',
            facet_row = "iter_inner",
            log_y=True,
            error_y="q_errors_p", 
            error_y_minus="q_errors_m", 
            template="plotly_white",
            height=1000)
fig3.update_layout(
    font_size = 20,
    width=1200, # in px
    height=900,
    )
# smaller linewidth
fig3.update_traces(
    selector=dict(),
    line_width=3,
    error_y_thickness = 0.3
)

fig3.update_xaxes(matches=None)
fig3.update_yaxes(matches=None)
fig3.update_xaxes(showticklabels=True)
fig3.update_yaxes(showticklabels=True)

#fig3.show()


# showing all
#fig.show()
#fig2.show()