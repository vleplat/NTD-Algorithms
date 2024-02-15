import numpy as np
from shootout.methods.runners import run_and_track
from shootout.methods.post_processors import df_to_convergence_df
import plotly.express as px
import pandas as pd
from mu_ntd.xps.audio_scripts.utils import frontiers_from_time_to_bar
from mu_ntd.xps.audio_scripts.utils import get_segmentation_from_txt
import mu_ntd.algorithms.Sparse_ntd as SNTD
import plotly.io as pio
import template_plot
pio.templates.default= "plotly_white+my_template"
#pio.kaleido.scope.mathjax = None  # stupid bug

# Command line input
# nb_seeds =
skip = 1

# The point here would be to compare extracted Q matrices with and without sparsity.
# We can toy with the 3d dimension to show sparsity helps fixing that dim.

variables = {
    "beta": 1,
    "weight": [0, 10, 100, 300, 500],  # 150, 200, 300],
    "ranks": [[32, 12, 10]],
    "init": "tucker",
    "itermax": 50,
    "iter_inner": 10,
    "tol": 0,
    "verbose_run": True,
    "scale": ["sinkhorn", "scalar", False],
    "epsilon": 1e-16,
    "seed": 1,  # no randomness
    "song": "come-together",  # or RWC1
    "song_path": "../Data/audio/",
    "xp": "sparse_Q",  # or sparse G
}

# Song Path
if variables["song"] == "RWC1":
    # Loading the NNlogMel Spectrogram tensor provided by Axel, 
    # as well as frontiers and bars in time scale
    tensor_spectrogram = np.load(variables["song_path"]+"1_nn_log_mel_grill_hop32_subdiv96.npy", allow_pickle = True) # RWC 1
    frontiers = get_segmentation_from_txt(variables["song_path"]+"RM-P001.BLOCKS.lab", "MIREX10")
    bars = np.load(variables["song_path"]+"1_bars.npy", allow_pickle=True)
elif variables["song"] == "come-together":
    tensor_spectrogram = np.load(variables["song_path"]+"tensor_spec_nnlm_come_together.npy", allow_pickle=True)  # Come together
    bars = np.load(variables["song_path"]+"come_together - bars.npy", allow_pickle=True)
    frontiers = get_segmentation_from_txt(variables["song_path"]+"The Beatles - Come Together.lab", "MIREX10")
# conversion
bars_time = bars
bars = [(bar[0], bar[1]) for bar in bars]
frontiers_time = frontiers
frontiers = [frontier[1] for frontier in frontiers]
# Getting the true bar index pos
frontiers_baridx = frontiers_from_time_to_bar(frontiers, bars)


@run_and_track(
    name_store="xp_audio"+variables["song"]+"-"+variables["xp"],
    path_store="./Results/",
    verbose=True,
    skip=skip,
    algorithm_names=["sNTD"],
    **variables)
def one_run(**cfg):
    # computing NTD
    if cfg["xp"] == "sparse_Q":
        l1weights = [0, 0, cfg["weight"], 0]
        l2weights = [cfg["weight"], cfg["weight"], 0, cfg["weight"]]
    elif cfg["xp"] == "sparse_G":
        l1weights=[0, 0, 0, cfg["weight"]]
        l2weights=[cfg["weight"], cfg["weight"], cfg["weight"], 0]
    core, factors, errors, timings, alpha, inner_cnt, sparsity = SNTD.sntd_mu(
            tensor_spectrogram,
            cfg["ranks"],
            l2weights=l2weights,
            l1weights=l1weights,
            init=cfg["init"],
            n_iter_max=cfg["itermax"],
            tol=cfg["tol"],
            beta=cfg["beta"],
            verbose=cfg["verbose_run"],
            return_costs=True,
            extrapolate=False,
            iter_inner=cfg["iter_inner"],
            accelerate=False,
            opt_rescale=cfg["scale"],
            epsilon=cfg["epsilon"])
    tol_sp = 2 * cfg["epsilon"]
    sparsity_Q = np.sum(factors[2] > tol_sp) / np.prod(np.shape(factors[2]))
    return {"errors":errors, "timings": timings, "factors": factors, "core": core, "sparsity_Q": sparsity_Q}


df = pd.read_pickle("Results/xp_audio"+variables["song"]+"-"+variables["xp"])
ovars = list(variables.keys())
df_conv = df_to_convergence_df(df, other_names=ovars, groups_names=ovars)
fig = px.line(df_conv, x="it", y="errors", color="scale", log_y=True, line_group="groups",facet_col="weight")
fig2 = px.line(
    df,
    x="weight",
    y="sparsity_Q",
    #line_group="groups",
    color="scale",
)
fig2.update_xaxes(type="category")


# Plotting script A La Axel
def reorder_rows(Q):
    # input: row matrix Q
    # output Q with permuted rows, so that first active rows are plotted first
    row_perm=[]
    n,m=Q.shape
    for j in range(m):
        imax = np.argmax(Q[:,j])
        if imax not in row_perm:
            row_perm.append(imax)
        if len(row_perm)==n:
            break
    # possible that some rows are not there, then we add them at the end
    if len(row_perm)<n:
        toadd = list(set([i for i in range(n)])-set(row_perm))
        # sort then by power
        norms = np.sum(Q[toadd,:], axis=1)
        order = np.argsort(norms)[::-1]
        row_perm = row_perm + [toadd[i] for i in order]
    return Q[row_perm,:]

def title_idx2(j, n):
    y = n-j-1
    if y%2==0:
        return y+1
    return y-1
def title_idx3(j, n):
    y = n-j-1
    if y%3==0:
        return y+2
    if y%3==1:
        return y
    return y-2

faclist = df["factors"]
Qlist = []
titles = [f"weight: {df['weight'][idx]} alg: {df['scale'][idx]}" for idx in range(len(df["factors"]))]
# Nonstandard normalization by rows of Q
for fac in faclist:
    Qt = fac[2].T/np.sum(fac[2],axis=1)
    Qt = reorder_rows(Qt)
    Qlist.append(Qt)
Qt3d = np.stack(Qlist, axis=2)
fig3 = px.imshow(Qt3d, facet_col=2, facet_col_wrap=3, aspect="auto", color_continuous_scale="Greys", facet_row_spacing=0.1)#, width=800, height=600) # titles how ?
# for speed debug
#frontiers_baridx = frontiers_baridx[:3]
for front in frontiers_baridx:
    # Extremely slow
    print("Slowly adding bar line in plotly imshows at index ",front)
    fig3.add_vline(x=front-0.5, line_color="green", line_width=1)
#fig3 = px.imshow(Qt3d, facet_col=2, facet_col_wrap=2, binary_string=True) # titles how ?
for i, title in enumerate(titles):
    #fig3.layout.annotations[title_idx2(i,len(Qlist))]['text'] = title
    # 3 conditions
    fig3.layout.annotations[title_idx3(i,len(Qlist))]['text'] = title

fig.update_layout(
    yaxis1=dict(title_text="Loss"),
    xaxis1=dict(title_text=""),
    xaxis2=dict(title_text=""),
    xaxis3=dict(title_text="iteration index"),
    xaxis4=dict(title_text=""),
    xaxis5=dict(title_text=""),
    legend=dict(title_text="balancing")
)
fig3.update_yaxes(
    showticklabels=False
)
fig3.update_layout(
    xaxis2=dict(title_text="Bar index"),
    yaxis7=dict(title_text="Pattern index")
)
fig.for_each_annotation(lambda a: a.update(text=a.text.replace("weight", "reg")))
fig3.for_each_annotation(lambda a: a.update(text=a.text.replace("weight", "reg")))

fig.show()
#fig2.show()
fig3.show()

fig.write_image("Results/xp_audio"+"-"+variables["song"]+"_loss.pdf")
fig3.write_image("Results/xp_audio"+"-"+variables["song"]+"_Q.pdf")