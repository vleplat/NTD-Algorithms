#!/usr/bin/env python
# coding: utf-8


# Python classics
import numpy as np
import tensorly as tl  # custom branch
from tensorly.decomposition import (
    non_negative_parafac_hals,
)  # cp hals implem with balancing

# import mu_ntd.algorithms.Sparse_ntd as SNTD  # TODO CP code beta div in mu_ntd
from mu_ntd.algorithms.utils import sparsify
import pandas as pd
import sys

# from mu_ntd.algorithms.sinkhorn import scale_factors_fro
from tensorly.solvers.penalizations import scale_factors_fro
from tlviz.factor_tools import factor_match_score as fms

# TODO:
# Debug rescale l2, seems to have an offset

# custom toolbox
# import shootout as sho
from shootout.methods.runners import run_and_track

# TODO shootout: write report with parameters
nb_seeds = int(sys.argv[1])
nb_seeds_init = int(sys.argv[2])
skip = False
if not nb_seeds:
    skip = True

# Configuration
variables = {
    "U_line": 30,
    "V_line": 29,
    "W_line": 28,
    "rank": [4],
    "rank_est": [6],
    "loss": "frobenius",  # "kl"
    "SNR": 80,
    "tol": 0,
    "n_iter_max": 30,  # TODO few iters?
    "epsilon": 1e-16,
    "beta": 1,  # TODO remove?
    "scale": [True, False],
    # "scale": ["optimal","scalar",False],
    "iter_inner": 30,
    "sparse_factors": False,
    # "weight" : [0,1e-1,5e-1,1,5,1e1,20,50,1e2,200,500,1e3], # for kl
    "weight": [1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1e-0, 5, 10, 100],
    "xp": ["sparse", "lra"],
    "accelerate": False,
    "init_scale": True,
    "unbalanced_tensor": False,
    "seed": [np.random.randint(1e5) for i in range(nb_seeds)],
    "seed_init": [np.random.randint(1e5) for i in range(nb_seeds_init)],
    "verbose_run": True,
}
name_store = "xp_scale_cp_Fro_tensorly" # gretsi replicate


@run_and_track(
    algorithm_names=["l1l2 HALS"],
    path_store="./Results/",
    name_store=name_store,
    add_track={"nbseed": nb_seeds},
    verbose=True,
    skip=skip,
    **variables,
)
def script_run(**cfg):
    # XP l1 or l2 everywhere
    match cfg["xp"]:
        case "sparse":
            l2weights = 3 * [0]  # cfg["weight"]]
            l1weights = 3 * [cfg["weight"]]
        case "lra":  # TODO fix sinkhorn??
            l2weights = 3 * [cfg["weight"]]
            l1weights = 3 * [0]  # cfg["weight"]]
    # XP sparse core
    # l2weights = 4*[cfg["weight"]]
    # l1weights = 4*[0]
    # l2weights[-1]=0 # mode 0 only
    # l1weights[-1]=cfg["weight"]

    # Seeding
    rng = np.random.RandomState(cfg["seed"] + hash("sNTD") % (2**32))
    # Generation of the input data tensor T
    factors_0 = []
    # deflation of components on U, V, W
    factors_0.append(rng.rand(cfg["U_line"], cfg["rank"]))
    factors_0.append(rng.rand(cfg["V_line"], cfg["rank"]))
    factors_0.append(rng.rand(cfg["W_line"], cfg["rank"]))
    if cfg["xp"] == "sparse":
        factors_0 = [sparsify(fac, 0.5) for fac in factors_0]
    # putting some weights
    if cfg["unbalanced_tensor"]:
        factors_0 = [
            factors_0[l] * np.array([10**i for i in range(cfg["rank"])])
            for l in range(3)
        ]
    # generate noise according to input SNR
    cp_true = tl.cp_tensor.CPTensor((None, factors_0))  # CP tensor
    cp_true.normalize()
    Ttrue = cp_true.to_tensor()
    N = rng.rand(cfg["U_line"], cfg["V_line"], cfg["W_line"])  # 1e-2
    sigma = 10 ** (-cfg["SNR"] / 20) * np.linalg.norm(Ttrue) / np.linalg.norm(N)
    T = Ttrue + sigma * N

    # Random initialization for the NTD
    rng2 = np.random.RandomState(cfg["seed_init"] + hash("sNTD") % (2**32))
    factors_init = []
    factors_init.append(rng2.rand(cfg["U_line"], cfg["rank_est"]))
    factors_init.append(rng2.rand(cfg["V_line"], cfg["rank_est"]))
    factors_init.append(rng2.rand(cfg["W_line"], cfg["rank_est"]))
    tensor_init = tl.cp_tensor.CPTensor((None, factors_init))

    # Initialization scaling --> breaks code ... TODO fix?
    if cfg["init_scale"]:
        tensor_init, rescale = scale_factors_fro(tensor_init, T, l1weights, l2weights)
        print(f"Initialization rescaled: {rescale}")
        # TODO add balancing

    # Solver parameters
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Call of solvers
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # ### Beta = 1 - MU no extrapolation no acceleration
    match cfg["loss"]:
        case "kl":
            # TODO
            print("not implemented")
            """
            core, factors, cost_fct_vals, toc, alpha, inner_cnt, sparsity = SNTD.sntd_mu(T, cfg["rank"], l2weights=l2weights, l1weights=l1weights, init = "custom", core_0 = tensor_init[0], factors_0 = tensor_init[1], n_iter_max = cfg["n_iter_max"], tol=cfg["tol"], beta = cfg["beta"],
                                                    fixed_modes = [], verbose = cfg["verbose_run"], return_costs = True, extrapolate=False, iter_inner=cfg["iter_inner"], accelerate=cfg["accelerate"], opt_rescale=cfg["scale"], epsilon=cfg["epsilon"])
            """

        case "frobenius":
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
                rescale=cfg["scale"],
                print_it=10,
            )
            # TODO REQ TIME, callback implem
            # TODO REQ SPARSITY
            toc = [i for i in range(len(cost_fct_vals))]  # its for now
            # sparsity = [[0]]

    # true_sparsity = tl.sum(core_0>0)/tl.prod(cfg["ranks_est"])
    # print(f"true sparsity {true_sparsity}")
    # print(f"core sparsity {sparsity[-1][-1]}")

    # Post-processing errors/FMS
    fms_score = fms(out_cp, cp_true)
    print(f"fms {fms_score}")

    # return {"errors": cost_fct_vals, "timings": toc, "sparsity_core": sparsity[-1], "error_final": cost_fct_vals[-1], "sparsity_final": sparsity[-1][-1],"target_sparsity": true_sparsity, "fms":fms_score}
    return {
        "errors": cost_fct_vals,
        "timings": toc,
        "error_final": cost_fct_vals[-1],
        "fms": fms_score,
    }


# Plotting
name_read = "Results/" + name_store
df = pd.read_pickle(name_read)
import plotly.express as px
import shootout.methods.post_processors as pp

# small tweaks to variables to adjust ranks #TODO adjust variables
# del variables[]
ovars = list(variables.keys())
# Convergence Plots
df_conv_it = pp.df_to_convergence_df(
    df, other_names=ovars, groups=True, groups_names=ovars, max_time=np.Inf
)
df_conv_time = pp.df_to_convergence_df(
    df, other_names=ovars, groups=True, groups_names=ovars
)  # , err_name="errors_interp", time_name="timings_interp", max_time=np.Inf)
# df_conv_sparse = pp.df_to_convergence_df(df, err_name="sparsity_core", other_names=ovars,groups=True,groups_names=ovars, max_time=np.Inf)

# Making plots
fig = px.line(
    df_conv_it,
    x="it",
    y="errors",
    color="scale",
    facet_col="weight",
    facet_row="xp",
    log_y=True,
    template="plotly_white",
    line_group="groups",
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

# fig5 = px.box(df, x="weight", y="sparsity_final", color="scale", log_x=True)
# fig5.update_xaxes(type='category')

fig6 = px.box(df, x="weight", y="fms", color="scale", log_x=True, facet_row="xp")
fig6.update_xaxes(type="category")

fig.show()
# fig2.show()
# fig3.show()
fig4.show()
# fig5.show()
fig6.show()

# time
