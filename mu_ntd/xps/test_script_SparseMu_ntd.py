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
    "SNR" : 100,
    "iter_inner" : 10
        }
#U_lines = [20,100],
#V_lines = [20,100],
#ranks=[[4,5,6],[10,2,2],[15,15,15]],
#sigma=[0,1e-2],
#iter_inner= [1,3,10])
name_store = "test_script_SparseMu_ntd_20122022"
@run_and_track(algorithm_names=["l1l2 MU", "l1l2 MU with HER"], path_store="./Results/",name_store=name_store,
                verbose=True, nb_seeds=2,**variables)
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
    Ttrue = tl.tenalg.multi_mode_dot(core_0, factors_0) 
    N = np.random.rand(U_lines, V_lines, W_lines) #1e-2
    sigma = 10**(-SNR/20)*np.linalg.norm(Ttrue)/np.linalg.norm(N) 
    T = Ttrue + sigma*N

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
    core, factors, cost_fct_vals, toc, alpha, inner_cnt, sparsity = SNTD.sntd_mu(T, ranks, l2weights=l2weight, l1weights=l1weight, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                                fixed_modes = [], verbose = False, return_costs = True, extrapolate=False, iter_inner=iter_inner, accelerate=False)
    # ### Beta = 1 - MU extrapolation and acceleration
    core_HER, factors_HER, cost_fct_vals_HER, toc_HER, alpha_HER, inner_cnt_HER, sparsity_HER = SNTD.sntd_mu(T, ranks, l2weights=l2weight, l1weights=l1weight, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                                fixed_modes = [], verbose = False, return_costs = True, extrapolate=True, iter_inner=iter_inner, accelerate=True)
    # normalize = 4*[None]
    #----------------------------------------------
    # Post-processing for checking identification
    #----------------------------------------------

    # normalisation
    for i in range(len(factors)):
        factors[i] = factors[i]/np.linalg.norm(factors[i],axis=0)
        factors_HER[i] = factors_HER[i]/np.linalg.norm(factors_HER[i],axis=0)
        factors_0[i] = factors_0[i]/np.linalg.norm(factors_0[i],axis=0)

    return {"errors": [cost_fct_vals, cost_fct_vals_HER], "timings": [toc,toc_HER]}#, "alpha":[alpha,alpha_HER], "congruence": [factors[2].T@factors_0[2],factors_HER[2].T@factors_0[2]]}


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
df = pp.interpolate_time_and_error(df, npoints = 100, adaptive_grid=True)
# 1. Convergence Plots
df_conv_it = pp.df_to_convergence_df(df,other_names=ovars,groups=True,groups_names=ovars, max_time=np.Inf)
df_conv_time = pp.df_to_convergence_df(df,other_names=ovars, groups=True,groups_names=ovars, err_name="errors_interp", time_name="timings_interp", max_time=np.Inf)
# renaming errors and time for convenience
df_conv_time = df_conv_time.rename(columns={"timings_interp": "timings", "errors_interp": "errors"})
# 2. Converting to median for iterations and timings
df_conv_median_it = pp.median_convergence_plot(df_conv_it, type="iterations")
df_conv_median_time = pp.median_convergence_plot(df_conv_time, type="timings")
# downsampling it by 5 for plotting (100 points per plot)
#df_conv_median_it = df_conv_median_it[df_conv_median_it["it"]%5==0]
# 3. Making plots
fig = px.line(df_conv_median_it, x="it", y="errors", color="sparse_data", log_y=True, facet_col="l2weight", facet_row="l1weight", error_y="q_errors_p", error_y_minus="q_errors_m", template="plotly_white") #sparsity? zip with extrapolate? will fail because groups are removed?
fig2 = px.line(df_conv_median_time, x="timings", y="errors", color="sparse_data", log_y=True, facet_col="l2weight", facet_row="l1weight", error_y="q_errors_p", error_y_minus="q_errors_m", template="plotly_white")
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
fig.update_xaxes(matches=None)
#fig.update_yaxes(matches=None)
fig.update_xaxes(showticklabels=True)
#fig.update_yaxes(showticklabels=True)
#fig2.update_xaxes(matches=None)
#fig2.update_yaxes(matches=None)
#fig2.update_xaxes(showticklabels=True)
#fig2.update_yaxes(showticklabels=True)
# time
