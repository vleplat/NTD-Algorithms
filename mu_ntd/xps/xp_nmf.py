#!/usr/bin/env python
# coding: utf-8


# Python classics
import numpy as np
import tensorly as tl  # custom branch
from tensorly.decomposition import (
    non_negative_parafac_hals,
)  # cp hals implem with balancing
from tensorly.solvers.penalizations import cp_opt_balance

import mu_ntd.algorithms.Sparse_ncp as SNCP
from mu_ntd.algorithms.utils import sparsify
import pandas as pd
import sys
import time
import json

# from mu_ntd.algorithms.sinkhorn import scale_factors_fro
from tensorly.solvers.penalizations import scale_factors_fro
from tlviz.factor_tools import factor_match_score as fms

# TODO:
# Debug rescale l2, seems to have an offset

# custom toolbox
# import shootout as sho
from shootout.methods.runners import run_and_track

nb_seeds = int(sys.argv[1])
nb_seeds_init = int(sys.argv[2])
skip = False
if not nb_seeds:
    skip = True

# Import Configuration
#file = open("./config/nmf_Fro.json")
#name_store = "xp_scale_nmf_Fro_dummy"
file = open("./config/nmf_KL.json")
name_store = "xp_nmf_paper"

variables = json.load(file)
# Tweaks for prototyping
#variables["weight"] = [0.01,0.02]
#variables["xp"]="sparse"
variables["seed"] = [np.random.randint(1e5) for i in range(nb_seeds)]
variables["seed_init"]= [np.random.randint(1e5) for i in range(nb_seeds_init)]


@run_and_track(
    algorithm_names=["l1 MU"],
    path_store="./Results/",
    name_store=name_store,
    add_track={"nbseed": nb_seeds},
    verbose=True,
    skip=skip,
    **variables,
)
def script_run(**cfg):
    # XP l1 + l2s or l2 everywhere
    if cfg["xp"] == "sparse":
        l2weights = 2*[0]#[0, cfg["weight"]]  # cfg["weight"]]
        l1weights = 2*[cfg["weight"]]
    elif cfg["xp"] == "indiv_sparse":
        l2weights = 2*[0]#[0, cfg["weight"]]  # cfg["weight"]]
        l1weights = [1, cfg["weight"]]
    elif cfg["xp"] == "lra":
        l2weights = 2*[cfg["weight"]]
        l1weights = 2*[0]  # cfg["weight"]]

    # sparsity detection
    tol = 2 * cfg["epsilon"]
    # Seeding
    rng = np.random.RandomState(cfg["seed"] + hash("sNTD") % (2**32))
    # Generation of the input data tensor T
    factors_0 = []
    # deflation of components on U, V, W
    factors_0.append(rng.rand(cfg["U_line"], cfg["rank"]))
    factors_0.append(rng.rand(cfg["V_line"], cfg["rank"]))
    if cfg["xp"] == "sparse" or cfg["xp"]=="indiv_sparse":
        print("sparsifying")
        factors_0 = [sparsify(fac, cfg["sparsify_coeff"]) for fac in factors_0]
    
    # generate noise according to input SNR
    cp_true = tl.cp_tensor.CPTensor((None, factors_0))  # CP tensor
    cp_true.normalize()
    Ttrue = cp_true.to_tensor()
    N = rng.rand(cfg["U_line"], cfg["V_line"])  # 1e-2
    sigma = 10 ** (-cfg["SNR"] / 20) * np.linalg.norm(Ttrue) / np.linalg.norm(N)
    T = Ttrue + sigma * N

    # Random initialization for the NCPD
    rng2 = np.random.RandomState(cfg["seed_init"] + hash("sNTD") % (2**32))
    factors_init = []
    # Unbalancing init
    factors_init.append(cfg["init_tilt"]*rng2.rand(cfg["U_line"], cfg["rank_est"]))
    factors_init.append(rng2.rand(cfg["V_line"], cfg["rank_est"]))

    scale_bool = bool(cfg["scale"])
    # TODO: factorize in function (maybe already in tensorly??)
    if cfg["scale"]=="init_only" and cfg["weight"]:
        print("Doing initialization scaling")
        for i in range(2):
            factors_init[i][factors_init[i]<=cfg["epsilon"]]=0
        for q in range(cfg["rank"]):
            thresh = tl.prod([tl.sum(tl.abs(factors_init[i][:, q])) for i in range(2)])
            if thresh == 0:
                for submode in range(2):
                    factors_init[submode][:, q] = 0
            else:
                regs = [
                    l1weights[i] * tl.sum(tl.abs(factors_init[i][:, q]))
                    + l2weights[i] * tl.norm(factors_init[i][:, q]) ** 2
                    for i in range(2)
                ]
                scales = cp_opt_balance(tl.tensor(regs), tl.tensor([1,1])) #for sparse xp only
                for submode in range(2):
                    factors_init[submode][:, q] = (
                        factors_init[submode][:, q] * scales[submode]
                    )
        for i in range(2):
            factors_init[i][factors_init[i]<=cfg["epsilon"]]=cfg["epsilon"]
    if cfg["scale"]=="init_only":
        scale_bool = False

    # Initialization scaling --> breaks code ... TODO fix?
    tensor_init = tl.cp_tensor.CPTensor((None, factors_init))
    if cfg["init_scale"]:
        # Note: In Gretsi, unbalanced updates also have no init rescaling
        # Quite unfair, it should be compared from the same rescaled init
        tensor_init, rescale = scale_factors_fro(tensor_init, T, l1weights, l2weights)
        print(f"Initialization rescaled: {rescale}")

    # callback
    TOC_NMF = []
    SCALES_W_NMF = []
    SPARSITY_W_NMF = []
    SCALES_H_NMF = []
    SPARSITY_H_NMF = []
    def time_tracer_nmf(cpdata,error):
        if cfg["xp"]=="sparse":
            SCALES_W_NMF.append(np.sum(cpdata[1][0][:,0]))
            SCALES_H_NMF.append(np.sum(cpdata[1][1][:,0]))
            SPARSITY_W_NMF.append(np.sum(cpdata[1][0]>tol)/(cfg["U_line"]*cfg["rank_est"]))
            SPARSITY_H_NMF.append(np.sum(cpdata[1][1]>tol)/(cfg["V_line"]*cfg["rank_est"]))
        elif cfg["xp"]=="indiv_sparse":
            SCALES_W_NMF.append(np.sum(cpdata[1][0][:,0]))
            SCALES_H_NMF.append(np.sum(cpdata[1][1][:,0]))
            SPARSITY_W_NMF.append(np.sum(cpdata[1][0]>tol)/(cfg["U_line"]*cfg["rank_est"]))
            SPARSITY_H_NMF.append(np.sum(cpdata[1][1]>tol)/(cfg["V_line"]*cfg["rank_est"]))
        elif cfg["xp"]=="lra":
            SCALES_W_NMF.append(tl.norm(cpdata[1][0][:,0])**2)
            SCALES_H_NMF.append(tl.norm(cpdata[1][1][:,0])**2)
        TOC_NMF.append(time.perf_counter())

    # Call of solvers
    # ### Beta = 1 - MU no extrapolation no acceleration
    if cfg["loss"] == "kl":
        out_cp, cost_fct_vals, toc, alpha, inner_cnt, sparsity = SNCP.sncp_mu(
            T,
            cfg["rank"],
            l2weights=l2weights,
            l1weights=l1weights,
            init="custom",
            factors_0=tensor_init[1],
            n_iter_max=cfg["n_iter_max"],
            tol=cfg["tol"],
            beta=cfg["beta"],
            fixed_modes=[],
            verbose=cfg["verbose_run"],
            return_costs=True,
            iter_inner=cfg["iter_inner"],
            accelerate=cfg["accelerate"],
            opt_rescale=scale_bool,
            epsilon=cfg["epsilon"],
            print_it=50,
        )

    elif cfg["loss"] == "frobenius":      
        out_cp, cost_fct_vals, _ = non_negative_parafac_hals(
            T,
            cfg["rank_est"],
            n_iter_max=cfg["n_iter_max"],
            init=tensor_init,
            tol=cfg["tol"],
            sparsity_coefficients=l1weights,
            ridge_coefficients=l2weights,
            verbose=cfg["verbose_run"],
            normalize_factors=False,
            return_errors=True,
            exact=False,
            inner_iter_max=cfg["iter_inner"],
            epsilon=cfg["epsilon"],
            rescale=scale_bool,
            print_it=10,
            callback=time_tracer_nmf
        )
        toc = [TOC_NMF[i]-TOC_NMF[0] for i in range(1,len(TOC_NMF))]#[i for i in range(len(cost_fct_vals))]  # its for now

    # Post-processing errors/FMS
    # First factor sparsity
    spfac_W = tl.sum(out_cp[1][0] > tol)/(cfg["U_line"]*cfg["rank_est"])
    spfac_H = tl.sum(out_cp[1][1] > tol)/(cfg["V_line"]*cfg["rank_est"])
    spfac_true = tl.sum(cp_true[1][0] > tol)/(cfg["U_line"]*cfg["rank_est"])
    # fms
    fms_score = fms(out_cp, cp_true)

    # Tracking low-rank things
    #out_cp.normalize()
    #print(out_cp[0])
    #Xhat = out_cp.to_tensor()
    #_,s,_ = np.linalg.svd(Xhat)
    #print(s[:6])
    #print(out_cp[1][0].T@out_cp[1][0])
    #print(out_cp[1][1].T@out_cp[1][1])

    return {
        "errors": cost_fct_vals,
        "timings": toc,
        "error_final": cost_fct_vals[-1],
        "fms": fms_score,
        "sparsity": spfac_W,
        "sparsity_H": spfac_H,
        "true_sparsity":spfac_true,
        "scale_W": SCALES_W_NMF[1:], # discarding init value
        "scale_H": SCALES_H_NMF[1:],
        "sp_W": SPARSITY_W_NMF[1:], 
        "sp_H": SPARSITY_H_NMF[1:], 
        "ratio_sp": spfac_W/spfac_H,
        "weights": out_cp[0], # to scout for pruned components
    }


# Plotting
name_read = "Results/" + name_store
df = pd.read_pickle(name_read)
import plotly.express as px
import shootout.methods.post_processors as pp

ovars = list(variables.keys())
# Convergence Plots
df_conv_it = pp.df_to_convergence_df(
    df, other_names=ovars, groups=True, groups_names=ovars, max_time=np.Inf
)

# Making plots
fig = px.line(
    df_conv_it,
    #x="it",
    x="timings",
    y="errors",
    color="scale",
    facet_col="weight",
    facet_row="xp",
    log_y=True,
    template="plotly_white",
    line_group="groups",
    title=name_store,
)
# fig2 = px.line(df_conv_time, x="timings", color="scale", facet_col="weight", y="errors", log_y=True, template="plotly_white", line_group="groups")
# core sparsity
# fig3 = px.line(df_conv_sparse, x="it", color="scale", facet_col="weight", y="sparsity_core", log_y=True, template="plotly_white", line_group="groups")
# smaller linewidth
fig.update_traces(selector=dict(), line_width=3, error_y_thickness=0.3)
# fig2.update_traces(
#    selector=dict(),
#    line_width=3,
#    error_y_thickness = 0.3
# )
fig.update_xaxes(matches=None)
fig.update_xaxes(showticklabels=True)

# final error wrt sparsity
fig4 = px.box(
    df,
    x="weight",
    y="error_final",
    color="scale",
    log_x=True,
    log_y=True,
    facet_row="xp",
)
fig4.update_xaxes(type="category")

fig5 = px.box(df, x="weight", y="sparsity", color="scale", facet_row="xp", log_x=True)
fig5.update_xaxes(type="category")

fig6 = px.box(df, x="weight", y="sparsity_H", color="scale", facet_row="xp", log_x=True)
fig6.update_xaxes(type="category")

fig7 = px.box(df, x="weight", y="fms", color="scale", log_x=True, facet_row="xp")
fig7.update_xaxes(type="category")

#fig7 = px.box(df, x="weight", y="ratio_sp", color="scale", log_x=True, facet_row="xp")
#fig7.update_xaxes(type="category")

fig.show()
# fig2.show()
# fig3.show()
fig4.show()
fig5.show()
fig6.show()
fig7.show()