#!/usr/bin/env python
# coding: utf-8

#----------------------------------------------------------------------
# Import usefull libraries
#----------------------------------------------------------------------
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import matplotlib
import scipy.io
# import ot
# import ot.plot
# from ot.datasets import make_1D_gauss as gauss
from astropy.io import fits
from scipy.io import savemat
import tensorly as tl
import mu_ntd.algorithms.Sparse_ntd as SNTD
import mu_ntd.algorithms.VCA as vca

#----------------------------------------------------------------------
# Function definition for visualization
#----------------------------------------------------------------------

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

#----------------------------------------------------------------------
# Load data set - HSI Samson
#----------------------------------------------------------------------
mat = scipy.io.loadmat('../../Data/JasperRidge/jasperRidge2_R198.mat')
A=mat['Y']
A = A/np.amax(A)
res1_array = mat['nCol']
res1 = res1_array[0][0]
res2_array = mat['nRow']
res2 = res2_array[0][0]
res3_array = mat['nBand']
res3 = 198


# Estimate endmembers with Vertex Component Analysis
Ae, indice, Yp = vca.vca(A,R=4,verbose = True, snr_input = 20)

counter_fig = 1
plt.figure(counter_fig)
plt.plot(Ae[:,0], color='blue', label='mat 1')
plt.plot(Ae[:,1], color='green', label='mat 2')
plt.plot(Ae[:,2], color='red', label='mat 3')
plt.plot(Ae[:,3], color='black', label='mat 4')
plt.xlabel('Spectral band')
plt.title('Columns of Ae')
plt.legend()
plt.show()

#  Reshape from matrix to tensor
T = np.reshape(A.T, (res1, res2, res3))
T = T/np.amax(T)
Ae.shape

# Visualization of input tensor T
light_jet = cmap_map(lambda x: x/2 + 0.5, matplotlib.cm.jet)
# nSlice = 80
# counter_fig = counter_fig + 1
# pl.figure(counter_fig)
# pl.imshow(T[:,:,nSlice].T, cmap=light_jet, aspect='auto')
# pl.title("Slice%1.0f" %nSlice+ " ")
# pl.colorbar()
# pl.show()


#----------------------------------------------------------------------
# Initialization
#----------------------------------------------------------------------

U_lines = res1
V_lines = res2
W_lines = res3
ranks = [80,80,4]


# Solver parameters
n_iter_max = 1700
beta = 1
iter_inner = 3
l2weight = [0, 50, 10, 5]            #(\mu_g, \mu_W, \mu_H, \mu_Q)    l2weight = [0, 0, 0, 0] 
l1weight = [10, 0, 0, 0]         #(\mu_g, \mu_W, \mu_H, \mu_Q)    l1weight = [25, 10, 10, 1]
verbose=False 
tol = 0 #running all iterations

# Random initialization for the NTD
factors_init = []
factors_init.append(np.random.rand(U_lines, ranks[0]))
factors_init.append(np.random.rand(V_lines, ranks[1]))
# factors_init.append(np.random.rand(W_lines, ranks[2]))
factors_init.append(Ae)
core_init = np.random.rand(ranks[0], ranks[1], ranks[2])


#----------------------------------------------------------------------
# Call of solvers
#----------------------------------------------------------------------

# Beta = 1 - MU no extrapolation no acceleration
core, factors, cost_fct_vals, toc, alpha, inner_cnt, sparsity = SNTD.sntd_mu(T, ranks, l2weights=l2weight, l1weights=l1weight, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                                fixed_modes = [], verbose = verbose, return_costs = True, extrapolate=False, iter_inner=iter_inner, accelerate=False)
# Beta = 1 - MU extrapolation and acceleration
# core_HER, factors_HER, cost_fct_vals_HER, toc_HER, alpha_HER, inner_cnt_HER, sparsity_HER = SNTD.sntd_mu(T, ranks, l2weights=l2weight, l1weights=l1weight, init = "custom", core_0 = core_init, factors_0 = factors_init, n_iter_max = n_iter_max, tol=tol, beta = beta,
                                                # fixed_modes = [], verbose = verbose, return_costs = True, extrapolate=True, iter_inner=iter_inner, accelerate=True, opt_rescale=False)



#----------------------------------------------------------------------
# Plot results
#----------------------------------------------------------------------
# Convergence plots
it1 = 1 # first iteration shown?
counter_fig = counter_fig + 1
plt.figure(counter_fig)
plt.semilogy(cost_fct_vals[it1:], color='blue', label='MU - Reba. on')
# plt.semilogy(cost_fct_vals_HER[it1:], color='black', label='MU - HER on')
plt.xlabel('Iteration number')
plt.ylabel('Objective function')
plt.title('Sparse beta-div NTD')
plt.legend()
plt.show()


counter_fig = counter_fig + 1
plt.figure(counter_fig)
plt.semilogy(toc[it1:],cost_fct_vals[it1:], color='blue', label='MU - Reba. on')
# plt.semilogy(toc_HER[it1:],cost_fct_vals_HER[it1:], color='black', label='MU - HER on')
plt.xlabel('CPU time')
plt.ylabel('Objective function')
plt.title('Sparse beta-div NTD')
plt.legend()
plt.show()


# Estimated endmembers - MU with reblancing
U = factors[0]
V = factors[1]
W = factors[2]
counter_fig = counter_fig + 1
plt.figure(counter_fig)
plt.plot(W[:,0], color='blue', label='mat 1')
plt.plot(W[:,1], color='green', label='mat 2')
plt.plot(W[:,2], color='red', label='mat 3')
plt.plot(W[:,3], color='black', label='mat 4')
plt.xlabel('Spectral band')
plt.title('Columns of W - MU + reba.')
plt.legend()
plt.show()


# Estimated endmembers - MU with HER
# U_HER = factors_HER[0]
# V_HER = factors_HER[1]
# W_HER = factors_HER[2]
# counter_fig = counter_fig + 1
# plt.figure(counter_fig)
# plt.plot(W_HER[:,0], color='blue', label='mat 1')
# plt.plot(W_HER[:,1], color='green', label='mat 2')
# plt.plot(W_HER[:,2], color='red', label='mat 3')
# plt.plot(W_HER[:,3], color='black', label='mat 4')
# plt.xlabel('Spectral band')
# plt.title('Columns of W - MU + HER')
# plt.legend()
# plt.show()



# Estimated abundance maps
mode = 2
A = tl.unfold(tl.tenalg.multi_mode_dot(core, factors, skip = mode), mode)

nMat = 0
A1 = np.reshape(A[nMat,:],(res1, res2))
nMat_graph = nMat + 1
counter_fig = counter_fig + 1
pl.figure(counter_fig)
pl.imshow(A1.T, cmap=light_jet, aspect='auto')
pl.title("Abundance map - mat %1.0f" %nMat_graph + " ")
pl.colorbar()
pl.show()

nMat = 1
A1 = np.reshape(A[nMat,:],(res1, res2))
nMat_graph = nMat + 1
counter_fig = counter_fig + 1
pl.figure(counter_fig)
pl.imshow(A1.T, cmap=light_jet, aspect='auto')
pl.title("Abundance map - mat %1.0f" %nMat_graph + " ")
pl.colorbar()
pl.show()

nMat = 2
A1 = np.reshape(A[nMat,:],(res1, res2))
nMat_graph = nMat + 1
counter_fig = counter_fig + 1
pl.figure(counter_fig)
pl.imshow(A1.T, cmap=light_jet, aspect='auto')
pl.title("Abundance map - mat %1.0f" %nMat_graph + " ")
pl.colorbar()
pl.show()

nMat = 3
A1 = np.reshape(A[nMat,:],(res1, res2))
nMat_graph = nMat + 1
counter_fig = counter_fig + 1
pl.figure(counter_fig)
pl.imshow(A1.T, cmap=light_jet, aspect='auto')
pl.title("Abundance map - mat %1.0f" %nMat_graph + " ")
pl.colorbar()
pl.show()



# # Estimated abundance maps
# mode = 2
# A = tl.unfold(tl.tenalg.multi_mode_dot(core_HER, factors_HER, skip = mode), mode)

# nMat = 0
# A1 = np.reshape(A[nMat,:],(res1, res2))
# nMat_graph = nMat + 1
# counter_fig = counter_fig + 1
# pl.figure(counter_fig)
# pl.imshow(A1.T, cmap=light_jet, aspect='auto')
# pl.title("Abundance map - mat %1.0f" %nMat_graph + " ")
# pl.colorbar()
# pl.show()

# nMat = 1
# A1 = np.reshape(A[nMat,:],(res1, res2))
# nMat_graph = nMat + 1
# counter_fig = counter_fig + 1
# pl.figure(counter_fig)
# pl.imshow(A1.T, cmap=light_jet, aspect='auto')
# pl.title("Abundance map - mat %1.0f" %nMat_graph + " ")
# pl.colorbar()
# pl.show()

# nMat = 2
# A1 = np.reshape(A[nMat,:],(res1, res2))
# nMat_graph = nMat + 1
# counter_fig = counter_fig + 1
# pl.figure(counter_fig)
# pl.imshow(A1.T, cmap=light_jet, aspect='auto')
# pl.title("Abundance map - mat %1.0f" %nMat_graph + " ")
# pl.colorbar()
# pl.show()

# fig, axs = pl.subplots(1, 3)
# nMat = 0
# A1 = np.reshape(A[nMat,:],(res1, res2))
# nMat_graph = nMat + 1
# axs[0].imshow(A1.T, cmap=light_jet, aspect='auto')
# axs[0].set_title("Abundance map - mat %1.0f" %nMat_graph + " ")
# axs[0].set_aspect('equal', 'box')
# nMat = 1
# A1 = np.reshape(A[nMat,:],(res1, res2))
# nMat_graph = nMat + 1
# axs[1].imshow(A1.T, cmap=light_jet, aspect='auto')
# axs[1].set_title("Abundance map - mat %1.0f" %nMat_graph + " ")
# axs[1].set_aspect('equal', 'box')
# nMat = 2
# A1 = np.reshape(A[nMat,:],(res1, res2))
# nMat_graph = nMat + 1
# axs[2].imshow(A1.T, cmap=light_jet, aspect='auto')
# axs[2].set_title("Abundance map - mat %1.0f" %nMat_graph + " ")
# axs[2].set_aspect('equal', 'box')
# #pl.colorbar()
# fig.tight_layout()
# fig.show()
