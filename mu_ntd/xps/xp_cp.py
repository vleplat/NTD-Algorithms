#!/usr/bin/env python
# coding: utf-8


# Python classics
import numpy as np
import tensorly as tl  # custom branch
from tensorly.decomposition import (
    non_negative_parafac_hals,
)  # cp hals implem with balancing

import mu_ntd.algorithms.Sparse_ncp as SNCP
from mu_ntd.algorithms.utils import sparsify
import pandas as pd
import sys
import time
import json

# from mu_ntd.algorithms.sinkhorn import scale_factors_fro
from tensorly.solvers.penalizations import scale_factors_fro, cp_opt_balance
from tlviz.factor_tools import factor_match_score as fms

# custom toolbox
# import shootout as sho
from shootout.methods.runners import run_and_track

nb_seeds = int(sys.argv[1])
nb_seeds_init = int(sys.argv[2])
skip = False
if not nb_seeds:
    skip = True

# Import Configuration
file = open("./config/cp_Fro.json")
name_store = "xp_cp_paper"

variables = json.load(file) 
variables["seed"]= [np.random.randint(1e5) for i in range(nb_seeds)]
variables["seed_init"]= [np.random.randint(1e5) for i in range(nb_seeds_init)]

@run_and_track(
    algorithm_names=["l2 HALS"],
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
        l2weights = [0, cfg["weight"] / 2, cfg["weight"] / 2]  # cfg["weight"]]
        l1weights = [cfg["weight"], 0, 0]
    elif cfg["xp"] == "lra":
        l2weights = 3 * [cfg["weight"]]
        l1weights = 3 * [0]  # cfg["weight"]]

    # Seeding
    rng = np.random.RandomState(cfg["seed"] + hash("sNTD") % (2**32))
    # Generation of the input data tensor T
    factors_0 = []
    # deflation of components on U, V, W
    factors_0.append(rng.rand(cfg["U_line"], cfg["rank"]))
    factors_0.append(rng.rand(cfg["V_line"], cfg["rank"]))
    factors_0.append(rng.rand(cfg["W_line"], cfg["rank"]))
    if cfg["xp"] == "sparse":
        factors_0 = [sparsify(fac, 0.3) for fac in factors_0]
    # generate noise according to input SNR
    cp_true = tl.cp_tensor.CPTensor((None, factors_0))  # CP tensor
    cp_true.normalize()
    Ttrue = cp_true.to_tensor()
    N = rng.rand(cfg["U_line"], cfg["V_line"], cfg["W_line"])  # 1e-2
    sigma = 10 ** (-cfg["SNR"] / 20) * np.linalg.norm(Ttrue) / np.linalg.norm(N)
    T = Ttrue + sigma * N

    # Random initialization for the NCPD
    rng2 = np.random.RandomState(cfg["seed_init"] + hash("sNTD") % (2**32))
    factors_init = []
    factors_init.append(rng2.rand(cfg["U_line"], cfg["rank_est"]))
    factors_init.append(rng2.rand(cfg["V_line"], cfg["rank_est"]))
    factors_init.append(rng2.rand(cfg["W_line"], cfg["rank_est"]))

    scale_bool = bool(cfg["scale"])
    # TODO: factorize in function (maybe already in tensorly??)
    if cfg["scale"]=="init_only" and cfg["weight"]:
        print("Doing initialization scaling")
        for i in range(3):
            factors_init[i][factors_init[i]<=cfg["epsilon"]]=0
        for q in range(cfg["rank"]):
            thresh = tl.prod([tl.sum(tl.abs(factors_init[i][:, q])) for i in range(2)])
            if thresh == 0:
                for submode in range(3):
                    factors_init[submode][:, q] = 0
            else:
                regs = [
                    l1weights[i] * tl.sum(tl.abs(factors_init[i][:, q]))
                    + l2weights[i] * tl.norm(factors_init[i][:, q]) ** 2
                    for i in range(3)
                ]
                scales = cp_opt_balance(tl.tensor(regs), tl.tensor([2,2,2]))
                for submode in range(3):
                    factors_init[submode][:, q] = (
                        factors_init[submode][:, q] * scales[submode]
                    )
        for i in range(3):
            factors_init[i][factors_init[i]<=cfg["epsilon"]]=cfg["epsilon"]
    if cfg["scale"]=="init_only":
        scale_bool = False

    tensor_init = tl.cp_tensor.CPTensor((None, factors_init))
    if cfg["init_scale"]:
        # Note: In Gretsi, unbalanced updates also have no init rescaling
        # Quite unfair, it should be compared from the same rescaled init
        tensor_init, rescale = scale_factors_fro(tensor_init, T, l1weights, l2weights, nonnegative=True)
        print(f"Initialization rescaled: {rescale}")

    # callback
    TOC_CP = []
    cost_fct_vals = []
    def time_tracer(cp_tensor,error):
        TOC_CP.append(time.perf_counter())
        cost_fct_vals.append(error)

    # Call of solvers
    # ### Beta = 1 - MU no extrapolation no acceleration
    if cfg["loss"] == "kl":
        out_cp, toc, cost_fct_vals, alpha, inner_cnt, sparsity = SNCP.sncp_mu(
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
            return_errors=True,
            verbose=cfg["verbose_run"],
            iter_inner=cfg["iter_inner"],
            accelerate=cfg["accelerate"],
            opt_rescale=scale_bool,
            epsilon=cfg["epsilon"],
            print_it=100,
        )

    elif cfg["loss"] == "frobenius":      
        out_cp = non_negative_parafac_hals(
            T,
            cfg["rank_est"],
            n_iter_max=cfg["n_iter_max"],
            init=tensor_init,
            tol=cfg["tol"],
            sparsity_coefficients=l1weights,
            ridge_coefficients=l2weights,
            verbose=cfg["verbose_run"],
            normalize_factors=False,
            inner_iter_max=cfg["iter_inner"],
            epsilon=cfg["epsilon"],
            rescale=scale_bool,
            print_it=10,
            callback=time_tracer
        )
        toc = [TOC_CP[i]-TOC_CP[0] for i in range(1,len(TOC_CP))]

    # Post-processing errors/FMS
    fms_score = fms(out_cp, cp_true)
    # First factor sparsity
    tol = 2 * cfg["epsilon"]
    spfac = tl.sum(out_cp[1][0] > tol)/(cfg["U_line"]*cfg["rank_est"])
    spfac_true = tl.sum(cp_true[1][0] > tol)/(cfg["U_line"]*cfg["rank_est"])

    out_cp.normalize()
    print(out_cp[0])

    return {
        "errors": cost_fct_vals,
        "timings": toc,
        "error_final": cost_fct_vals[-1],
        "fms": fms_score,
        "sparsity": spfac,
        "true_sparsity": spfac_true,
        "weights": out_cp[0], # to scout for pruned components
        "component_count": sum(out_cp[0]>1e3*tol)
    }


# Plotting
name_read = "Results/" + name_store
df = pd.read_pickle(name_read)
import plotly.express as px
import shootout.methods.post_processors as pp
import plotly.io as pio
import template_plot
# template usage
pio.templates.default= "plotly_white+my_template"

ovars = list(variables.keys())
# Convergence Plots
#df_conv_it = pp.df_to_convergence_df(
    #df, other_names=ovars, groups=True, groups_names=ovars, max_time=np.Inf
#)
#df_conv_time = pp.df_to_convergence_df(
    #df, other_names=ovars, groups=True, groups_names=ovars
#)  # , err_name="errors_interp", time_name="timings_interp", max_time=np.Inf)
# df_conv_sparse = pp.df_to_convergence_df(df, err_name="sparsity_core", other_names=ovars,groups=True,groups_names=ovars, max_time=np.Inf)

## Making plots
#fig = px.line(
    #df_conv_it,
    #x="it",
    ##x="timings",
    #y="errors",
    #color="scale",
    #facet_col="weight",
    #facet_row="xp",
    #log_y=True,
    #template="plotly_white",
    #line_group="groups",
    #title=name_store,
#)
## fig2 = px.line(df_conv_time, x="timings", color="scale", facet_col="weight", y="errors", log_y=True, template="plotly_white", line_group="groups")
## core sparsity
## fig3 = px.line(df_conv_sparse, x="it", color="scale", facet_col="weight", y="sparsity_core", log_y=True, template="plotly_white", line_group="groups")
## smaller linewidth
#fig.update_traces(selector=dict(), line_width=3, error_y_thickness=0.3)
## fig2.update_traces(
##    selector=dict(),
##    line_width=3,
##    error_y_thickness = 0.3
## )
#fig.update_xaxes(matches=None)
#fig.update_xaxes(showticklabels=True)

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

#fig5 = px.box(df, x="weight", y="sparsity", color="scale", facet_row="xp", log_x=True)
#fig5.update_xaxes(type="category")

fig6 = px.box(df, x="weight", y="fms", color="scale", log_x=True, facet_row="xp")
fig6.update_xaxes(type="category")

fig7 = px.box(df, x="weight", y="component_count", color="scale", log_x=True, facet_row="xp")
fig7.update_xaxes(type="category")


# time
for fi in [fig4,fig6,fig7]:
    fi.update_layout(
        xaxis=dict(title_text="Regularization parameter"),
        legend=dict(title_text="balancing")
    )


fig4.show()
fig6.show()
fig7.show()

fig4.write_image("Results/"+name_store+"_loss.pdf")
fig6.write_image("Results/"+name_store+"_fms.pdf")
fig7.write_image("Results/"+name_store+"_components.pdf")


