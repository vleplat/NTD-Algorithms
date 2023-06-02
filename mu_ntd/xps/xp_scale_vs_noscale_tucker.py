#!/usr/bin/env python
# coding: utf-8


# Python classics
import numpy as np
import tensorly as tl # custom branch
from tensorly.decomposition import non_negative_tucker_hals # Tucker hals implem with balancing
from tensorly import tucker_tensor
import mu_ntd.algorithms.Sparse_ntd as SNTD
from mu_ntd.algorithms.utils import sparsify, tucker_fms
import matplotlib.pyplot as plt
import pandas as pd
import sys
from mu_ntd.algorithms.sinkhorn import scale_factors_fro

# TODO: 
# Debug rescale l2, seems to have an offset

# custom toolbox
#import shootout as sho
from shootout.methods.runners import run_and_track

# todo shootout: write report with parameters
nb_seeds = int(sys.argv[1])
nb_seeds_init = int(sys.argv[2])
skip=False
if not nb_seeds:
    skip=True

# Configuration
variables={    
    #"weight" : [0,1e-1,5e-1,1,5,1e1,20,50,1e2,200,500,1e3], # for kl
    "weight" : [1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1.0], # no low-rank effect :( ?? mb bug ?
    #"weight" : [1e-6],
    "xp" :  ["lra",  "sparse"],
    "scale": ["sinkhorn", "scalar", False],
    #"seed": [np.random.randint(1e5) for i in range(nb_seeds)],
    "seed": [i+15 for i in range(nb_seeds)],
    #"seed_init": [np.random.randint(1e5) for i in range(nb_seeds_init)],
    "seed_init": [i+20 for i in range(nb_seeds_init)],
    "loss": "kl",#"frobenius", #"kl"
    "U_line" : 20,
    "V_line" : 19,
    "W_line" : 18,
    "ranks" : [[4,3,2]],
    "ranks_est": [[4,3,4]],
    "SNR" : 80,
    "tol" : 0,
    "n_iter_max" : 300,
    "epsilon": 1e-16,
    "beta" : 1,
    #"scale": ["scalar",False],
    "iter_inner" : 30,
    "sparse_factors" : True,
    "sparse_core" : True,
    "accelerate" : False,
    "init_scale" : True, #TODO check if it does not break
    "unbalanced_tensor": False,
    "verbose_run": True,
    }
name_store = "xp_scale_tucker_kl"

@run_and_track(algorithm_names=["l1l2 MU"], path_store="./Results/",name_store=name_store, add_track={"nbseed": nb_seeds}, verbose=True, skip=skip, **variables)
def script_run(**cfg):
    match cfg["xp"]:
        case "sparse":
            l1weights = 4*[cfg["weight"]]
            l2weights = 4*[0]#cfg["weight"]]
        case "lra": #TODO fix sinkhorn??
            l1weights = 4*[0]#cfg["weight"]]
            l2weights = 4*[cfg["weight"]]
    # XP sparse core
    #l2weights = 4*[cfg["weight"]]
    #l1weights = 4*[0]
    #l2weights[-1]=0 # mode 0 only
    #l1weights[-1]=cfg["weight"]

    # Seeding 
    rng = np.random.RandomState(cfg["seed"]+hash("sNTD")%(2**32))
    # Generation of the input data tensor T
    factors_0 = []
    # deflation of components on U, V, W
    factors_0.append(rng.rand(cfg["U_line"], cfg["ranks"][0]))
    factors_0.append(rng.rand(cfg["V_line"], cfg["ranks"][1]))
    factors_0.append(rng.rand(cfg["W_line"], cfg["ranks"][2]))
    if cfg["sparse_factors"]:
        factors_0 = [sparsify(fac,0.5) for fac in factors_0]
    core_0 = rng.rand(cfg["ranks"][0], cfg["ranks"][1], cfg["ranks"][2])
    if cfg["sparse_core"]:
        core_0 = sparsify(core_0,0.5)
    # putting some weights
    if cfg["unbalanced_tensor"]:
        factors_0 = [factors_0[l] * np.array([10**i for i in range(cfg["ranks"][l])]) for l in range(3)]
    tucker_true = tucker_tensor.TuckerTensor((core_0, factors_0))
    tucker_true.normalize()
    # generate noise according to input SNR
    Ttrue = tucker_true.to_tensor()
    N = rng.rand(cfg["U_line"], cfg["V_line"], cfg["W_line"]) #1e-2
    sigma = 10**(-cfg["SNR"]/20)*np.linalg.norm(Ttrue)/np.linalg.norm(N) 
    T = Ttrue + sigma*N

    # Random initialization for the NTD
    rng2 = np.random.RandomState(cfg["seed_init"]+hash("sNTD")%(2**32))
    factors_init = []
    factors_init.append(rng2.rand(cfg["U_line"], cfg["ranks_est"][0]))
    factors_init.append(rng2.rand(cfg["V_line"], cfg["ranks_est"][1]))
    factors_init.append(rng2.rand(cfg["W_line"], cfg["ranks_est"][2]))
    core_init = rng2.rand(cfg["ranks_est"][0], cfg["ranks_est"][1], cfg["ranks_est"][2])
    tensor_init = tucker_tensor.TuckerTensor((core_init,factors_init))


    # Initialization scaling (optimized for l2 loss)
    if cfg["init_scale"]:
        tensor_init, rescale = scale_factors_fro(tensor_init, T, l1weights, l2weights, format_tensor="tucker")
        print(f"Initialization rescaled: {rescale}")

    # Solver parameters
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Call of solvers
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


    # ### Beta = 1 - MU no extrapolation no acceleration
    match cfg["loss"]:
        case "kl":
            core, factors, cost_fct_vals, toc, alpha, inner_cnt, sparsity = SNTD.sntd_mu(T, cfg["ranks"], l2weights=l2weights, l1weights=l1weights, init = "custom", core_0 = tensor_init[0], factors_0 = tensor_init[1], n_iter_max = cfg["n_iter_max"], tol=cfg["tol"], beta = cfg["beta"],
                                                    fixed_modes = [], verbose = cfg["verbose_run"], return_costs = True, extrapolate=False, iter_inner=cfg["iter_inner"], accelerate=cfg["accelerate"], opt_rescale=cfg["scale"], epsilon=cfg["epsilon"])
    
        case "frobenius":
            out_tucker, cost_fct_vals = non_negative_tucker_hals(T, cfg["ranks"], n_iter_max=cfg["n_iter_max"], init=tensor_init, tol=cfg["tol"],sparsity_coefficients=l1weights, ridge_coefficients=l2weights, verbose=cfg["verbose_run"], normalize_factors=False, return_errors=True, exact=False,    algorithm="fista", inner_iter_max = cfg["iter_inner"], #todo implement, or have some option dictionary?
            epsilon = cfg["epsilon"], rescale=cfg["scale"], print_it=10)
            core = out_tucker[0]
            factors = out_tucker[1]
            # TODO REQ TIME, callback implem
            # TODO REQ SPARSITY
            toc = [i for i in range(len(cost_fct_vals))] # its for now
            #sparsity = [[0]]

    #true_sparsity = tl.sum(core_0>0)/tl.prod(cfg["ranks_est"])
    #print(f"true sparsity {true_sparsity}")
    #print(f"core sparsity {sparsity[-1][-1]}")

    # Post-processing errors/FMS
    # TODO: error, getting 1 with crap ?
    fms_score, perms = tucker_fms((core,factors),(core_0,factors_0))
    print(f"fms {fms_score, perms}")

    #return {"errors": cost_fct_vals, "timings": toc, "sparsity_core": sparsity[-1], "error_final": cost_fct_vals[-1], "sparsity_final": sparsity[-1][-1],"target_sparsity": true_sparsity, "fms":fms_score}
    return {"errors": cost_fct_vals, "timings": toc, "error_final": cost_fct_vals[-1], "fms":fms_score}



# Plotting
name_read = "Results/"+name_store
df = pd.read_pickle(name_read)
import plotly.express as px
import shootout.methods.post_processors as pp
import shootout.methods.plotters as pt

# small tweaks to variables to adjust ranks #TODO adjust variables
#del variables[]
ovars = list(variables.keys())
# Convergence Plots
df_conv_it = pp.df_to_convergence_df(df,other_names=ovars,groups=True,groups_names=ovars, max_time=np.Inf)
df_conv_time = pp.df_to_convergence_df(df,other_names=ovars, groups=True,groups_names=ovars)#, err_name="errors_interp", time_name="timings_interp", max_time=np.Inf)
#df_conv_sparse = pp.df_to_convergence_df(df, err_name="sparsity_core", other_names=ovars,groups=True,groups_names=ovars, max_time=np.Inf)

# Making plots
fig = px.line(df_conv_it, x="it", y="errors", color="scale", facet_col="weight", log_y=True, template="plotly_white", line_group="groups", facet_row="xp")
#fig2 = px.line(df_conv_time, x="timings", color="scale", facet_col="weight", y="errors", log_y=True, template="plotly_white", line_group="groups")
# core sparsity
#fig3 = px.line(df_conv_sparse, x="it", color="scale", facet_col="weight", y="sparsity_core", log_y=True, template="plotly_white", line_group="groups")
# smaller linewidth
fig.update_traces(
    selector=dict(),
    line_width=3,
    error_y_thickness = 0.3
)
#fig2.update_traces(
#    selector=dict(),
#    line_width=3,
#    error_y_thickness = 0.3
#)
fig.update_xaxes(matches=None)
fig.update_xaxes(showticklabels=True)

# final error wrt sparsity
fig4 = px.box(df, x="weight", y="error_final", color="scale", log_x=True, log_y=True, facet_row="xp")
fig4.update_xaxes(type='category')

#fig5 = px.box(df, x="weight", y="sparsity_final", color="scale", log_x=True)
#fig5.update_xaxes(type='category')

fig6 = px.box(df, x="weight", y="fms", color="scale", log_x=True, facet_row="xp")
fig6.update_xaxes(type='category')

fig.show()
#fig2.show()
#fig3.show()
fig4.show()
#fig5.show()
fig6.show()

# time
