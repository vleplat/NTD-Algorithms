#!/usr/bin/env python
# coding: utf-8


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
nb_seeds = 2

variables={    
    "U_lines" : 20,
    "V_lines" : 20,
    "ranks" : [[4,5,6]],
    "SNR" : 80,
    "scale": [True,False],
    "iter_inner" : 50,
    "sparse_factors" : [False],
    "sparse_core" : [True],
    "l1weight" : [0.5,2.5],
    }
name_store = "xp_scale_13-02-2023"
@run_and_track(algorithm_names=["l1l2 MU", "l1l2 MU with HER"], path_store="./Results/",name_store=name_store,
                verbose=True, nb_seeds=nb_seeds,**variables, seeded_fun=True)
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
    l1weight = 0,
    l2weight = 0,
    verbose=False,
    scale=True,
    accelerate=False,
    sparse_factors = False,
    sparse_core = False,
    seed=5
    ):
    # weights processings
    #l2weights = [l2weight, l2weight, l2weight, 0]
    l2weights = [l1weight, l1weight, l1weight, 0]
    l1weights = [0, 0, 0, l1weight]
    #if l1weights==0:
        #l2weights[-1] = l2weights[0] #setting ridge on core if no sparsity but ridge on factors
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


    # ### Beta = 1 - MU no extrapolation no acceleration
    core, factors, cost_fct_vals, toc, alpha, inner_cnt, sparsity = SNTD.sntd_mu(T, ranks, l2weights=l2weights, l1weights=l1weights, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                                fixed_modes = [], verbose = verbose, return_costs = True, extrapolate=False, iter_inner=iter_inner, accelerate=accelerate, opt_rescale=scale)
    # ### Beta = 1 - MU extrapolation and acceleration
    core_HER, factors_HER, cost_fct_vals_HER, toc_HER, alpha_HER, inner_cnt_HER, sparsity_HER = SNTD.sntd_mu(T, ranks, l2weights=l2weights, l1weights=l1weights, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                                fixed_modes = [], verbose = verbose, return_costs = True, extrapolate=True, iter_inner=iter_inner, accelerate=accelerate, opt_rescale=scale)
    # normalize = 4*[None]
    #----------------------------------------------
    # Post-processing for checking identification
    #----------------------------------------------

    # normalisation
    for i in range(len(factors)):
        factors[i] = factors[i]/np.linalg.norm(factors[i],axis=0)
        factors_HER[i] = factors_HER[i]/np.linalg.norm(factors_HER[i],axis=0)
        factors_0[i] = factors_0[i]/np.linalg.norm(factors_0[i],axis=0)

    return {"errors": [cost_fct_vals, cost_fct_vals_HER], "timings": [toc,toc_HER], "alpha":[alpha,alpha_HER], "sparsity_core": [sparsity[-1],sparsity_HER[-1]]}


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
#df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True)
# 1. Convergence Plots
df_conv_it = pp.df_to_convergence_df(df,other_names=ovars,groups=True,groups_names=ovars, max_time=np.Inf)
df_conv_time = pp.df_to_convergence_df(df,other_names=ovars, groups=True,groups_names=ovars)#, err_name="errors_interp", time_name="timings_interp", max_time=np.Inf)
df_conv_sparse = pp.df_to_convergence_df(df, err_name="sparsity_core", other_names=ovars,groups=True,groups_names=ovars, max_time=np.Inf)
# renaming errors and time for convenience
#df_conv_time = df_conv_time.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
# 2. Converting to median for iterations and timings
#df_conv_median_it = pp.median_convergence_plot(df_conv_it, type="iterations")
#df_conv_median_time = pp.median_convergence_plot(df_conv_time, type="timings")
# downsampling it by 5 for plotting (100 points per plot)
#df_conv_median_it = df_conv_median_it[df_conv_median_it["it"]%5==0]
# 3. Making plots
fig = px.line(df_conv_it, x="it", y="errors", color="algorithm", facet_col="scale", facet_row="l1weight", log_y=True, template="plotly_white", line_group="groups")
figbis = px.line(df_conv_it, x="it", y="errors", color="scale", facet_col="algorithm", facet_row="l1weight", log_y=True, template="plotly_white", line_group="groups")
fig2 = px.line(df_conv_time, x="timings", color="algorithm", facet_col="scale", facet_row="l1weight", y="errors", log_y=True, template="plotly_white", line_group="groups")
# core sparsity
fig3 = px.line(df_conv_sparse, x="it", color="algorithm", facet_col="scale", facet_row="l1weight", y="sparsity_core", log_y=True, template="plotly_white", line_group="groups")
# smaller linewidth
fig.update_traces(
    selector=dict(),
    line_width=3,
    error_y_thickness = 0.3
)
figbis.update_traces(
    selector=dict(),
    line_width=3,
    error_y_thickness = 0.3
)
fig2.update_traces(
    selector=dict(),
    line_width=3,
    error_y_thickness = 0.3
)
fig.update_xaxes(matches=None)
fig.update_xaxes(showticklabels=True)
#fig.update_yaxes(showticklabels=True)
#fig2.update_xaxes(matches=None)
#fig2.update_yaxes(matches=None)
#fig2.update_xaxes(showticklabels=True)
#fig2.update_yaxes(showticklabels=True)
fig.show()
figbis.show()
fig2.show()
fig3.show()

# time
