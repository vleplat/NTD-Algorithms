import tensorly as tl
import numpy as np
import copy

# Testing tensor sinkhorn for the core

def tensor_sinkhorn(tensor, marginals, itermax=5):
    """only works for 1d marginals for now
    Not used in Tucker.

    Requires feasible marginals

    Parameters
    ----------
    tensor : tensor
        _description_
    marginals : list of 1D tensor
        _description_
    """
    dims = tl.shape(tensor)
    tensor_scaled = tl.copy(tl.abs(tensor)) # we rescale iteratively: cheaper but sometimes more memory costly
    scales = [tl.ones(dim) for dim in dims] # stores the scales for each mode
    # final marginals
    marginals_obtained = []
    for mode in range(tensor.ndim):
        marginals_obtained.append(tl.sum(tensor_scaled,axis=tuple([i for i in range(tensor.ndim) if i!=mode])))
    # some cost function
    loss = sum([abs(tl.sum(marginals_obtained[mode] - marginals[mode])) for mode in range(tensor.ndim)])
    print(f"iteration {-1} loss {loss}")
    for it in range(itermax):
        for mode in range(tensor.ndim):
            current_marginal_tensor = tl.sum(tensor_scaled,axis=tuple([i for i in range(tensor.ndim) if i!=mode]), keepdims=True)
            scaling = current_marginal_tensor/marginals[mode].reshape(current_marginal_tensor.shape) # for broadcasting
            tensor_scaled = tensor_scaled/scaling # correct marginals along mode mode
            scales[mode] = scales[mode]*scaling.reshape(dims[mode])
        # final marginals
        marginals_obtained = []
        for mode in range(tensor.ndim):
            marginals_obtained.append(tl.sum(tensor_scaled,axis=tuple([i for i in range(tensor.ndim) if i!=mode])))
        # some cost function
        loss = sum([abs(tl.sum(marginals_obtained[mode] - marginals[mode])) for mode in range(tensor.ndim)])
        print(f"iteration {it} loss {loss}")

        
    return tensor_scaled, scales, marginals_obtained

def tensor_online_sinkhorn(tensor, regs, lamb_g, hom_g=1, itermax=10, verbose=False):
    """
    Solves the regularization scaling problem, similar to Sinkhorn but with unknown marginals learnt on the fly.

    Only need the regularizations for each factor instead of the factors as input.

    TODO: care about 0 scale

    Parameters
    ----------
    tensor : tensor, required
        _description_
    regs : list of 2d tensors, required
        _description_
    lamb_g : float, required
        regularization parameter for the core 
    hom_g : int
        homogeneity degree of reg_g, by default 1
    itermax : int
        number of iterations for the scaling, default 10
    """
    if hom_g==1:
        reg_g = lambda x: tl.abs(x)
    elif hom_g==2:
        reg_g = lambda x: x**2
    else:
        print("hom_g not 1 or 2 not implemented")

    # Precomputations
    dims = tl.shape(tensor)
    nmodes = tensor.ndim
    tensor_scaled = tl.copy(tensor) # we rescale iteratively, core is small so its ok
    # final marginals
    if verbose:
        loss = sum([sum(reg) for reg in regs])+lamb_g*tl.sum(reg_g(tensor_scaled))
        print(f"iteration {-1} loss {loss}")

    # initial marginals
    marginals = tl.copy(regs) # marginals of reg_g(tensor)
    scalings = [1 for i in range(nmodes)]

    for it in range(itermax): #TODO check lamda_g
        for mode in range(nmodes):
            tensor_marginal_mode = lamb_g*tl.sum(reg_g(tensor_scaled), axis=tuple([i for i in range(nmodes) if i!=mode]), keepdims=True)
            marginals[mode] = tl.sqrt(marginals[mode]*tensor_marginal_mode.reshape(dims[mode])) # Vector or marginals over mode U, also norm of U and G
            # Marginals will be zero if one of fac or core has zero marginal. We put ones where there are zeros in the tensor marginals to have 0/1=0.
            tensor_marginal_mode[tensor_marginal_mode==0] = 1
            scale = marginals[mode].reshape(tensor_marginal_mode.shape)/tensor_marginal_mode
            scalings[mode] *=  scale.reshape(dims[mode])
            tensor_scaled = tensor_scaled*(scale**(1/hom_g)) # correct marginals along mode mode
        # some cost function
        if verbose:
            loss = sum([sum(marginals[i]) for i in range(nmodes)]+[lamb_g*tl.sum(reg_g(tensor_scaled))])
            print(f"iteration {it} loss {loss}")
            print(f"consistency {np.std([sum(mar) for mar in marginals])}")
    # Recompute all marginals at the end (not necessary) for feasible output
    for mode in range(nmodes):
        marginals[mode] = tl.sum(reg_g(tensor_scaled), axis=tuple([i for i in range(nmodes) if i!=mode]))
    if verbose:
        print(f"consistency {np.std([sum(mar) for mar in marginals])}")

    return tensor_scaled, scalings, marginals

def tucker_implicit_sinkhorn_balancing(factors, core, regs, lamb_g, hom_reg, itermax=10):
    """A one liner to balance factors and core using adaptive sinkhorn.

    Parameters
    ----------
    factors : list of arrays, required
        factors of the Tucker model
    core : tl tensor, required 
        core of the Tucker model
    regs : list of floats, required
        the list of regularization values for the input factors. May contain zeroes.
    hom_reg : list of ints, required
        homogeneity degrees for the factors regularizations, and the core
    lamb_g : float, required
        regularization parameter for the core penalization
    hom_g : int, optional
        homogeneity degree for the core regularization, by default 1
    itermax : int, optional
        maximal number of scaling iterations, by default 10

    Returns
    -------
    factors, core : scaled factors and core
    scales : list of lists with the scaling of the core/factors on each mode
    """

    core, scales, _ = tensor_online_sinkhorn(core, regs, lamb_g=lamb_g, hom_g=hom_reg[-1], itermax=itermax)
    for mode in range(core.ndim):
        # look for zero scales
        a = tl.tensor(scales[mode]>0)
        scales[mode][scales[mode]==0]=1
        #print(f"debug: {scales}\n, a {a}")
        # put ones in there, factors will be scaled to zero
        rescale_fac = a/scales[mode]
        factors[mode] *= rescale_fac[None,:]**(1/hom_reg[mode])
        scales[mode] = scales[mode]*a #input true 0 instead of 1
    return factors, core, scales

def opt_scaling(regs, hom_deg):
    '''
    Computes the multiplicative constants to scale factors such that regularizations are balanced.
    The problem solved is 
        min_{a_i} \sum a_i s.t.  \prod a_i^{p_i}=q
    where a_i = regs[i] and p_i = hom_deg[i]

    This is suboptimal since it will only scale factors and core globally (not columnwise), but is fast because in closed form.

    Parameters
    ----------
    regs: 1d np array
        the input regularization values
    hom_deg: 1d numpy array
        homogeneity degrees of each regularization term

    Returns
    -------
    scales: list of floats
        the scale to balance the factors. Its product should be one (scale invariance).
    '''
    # 1. compute q
    prod_q = np.prod(regs**(1/hom_deg))

    # 2. compute beta
    beta = (prod_q*np.prod(hom_deg**(1/hom_deg)))**(1/np.sum(1/hom_deg))

    # 3. compute scales
    scales = [(beta/regs[i]/hom_deg[i])**(1/hom_deg[i]) for i in range(len(regs))]
    
    return scales

def tucker_implicit_scalar_balancing(factors, core, regs, hom_deg):
    """A one liner to balance factors and core using scalar scaling.

    Parameters
    ----------
    factors : _type_
        _description_
    core : _type_
        _description_
    regs : _type_
        _description_
    hom_deg : _type_
        _description_
    """    """
    """
    scales = opt_scaling(np.array(regs),np.array(hom_deg))
    for mode in range(tl.ndim(core)):
        factors[mode] *= scales[mode]
    core = core*scales[-1]

    return factors, core, scales

def scale_factors_fro(tensor,data,sparsity_coefficients,ridge_coefficients, format_tensor="cp"):
    '''
    Optimally scale [G;A,B,C] in 
    
    min_x \|data - x^{n_modes} [G;A_1,A_2,A_3]\|_F^2 + \sum_i sparsity_coefficients_i \|A_i\|_1 + \sum_j ridge_coefficients_j \|A_j\|_2^2

    This avoids a scaling problem when starting the separation algorithm, which may lead to zero-locking.
    The problem is solved by finding the positive roots of a polynomial.

    Works with any number of modes and both CP and Tucker, as specified by the `format` input. For "tucker" format, sparsity and ridge have an additional final value for the core reg.
    '''
    factors = copy.deepcopy(tensor[1])
    if format_tensor=="tucker":
        factors.append(tensor[0])
    n_modes = len(factors)
    l1regs = [sparsity_coefficients[i]*tl.sum(tl.abs(factors[i])) for i in range(n_modes)]
    l2regs= [ridge_coefficients[i]*tl.norm(factors[i])**2 for i in range(n_modes)]
    # We define a polynomial
    # a x^{2n_modes} + b x^{n_modes} + c x^{2} + d x^{1}
    # and find the roots of its derivative, compute the value at each one, and return the optimal scale x and scaled factors.
    a = tensor.norm()**2
    b = -2*tl.sum(data*tensor.to_tensor())
    c = sum(l2regs)
    d = sum(l1regs)
    poly = [0 for i in range(2*n_modes+1)]
    poly[1] = d
    poly[2] = c
    poly[n_modes] = b
    poly[2*n_modes] = a
    poly.reverse()
    grad_poly = [0 for i in range(2*n_modes)]
    grad_poly[0] = d
    grad_poly[1] = 2*c
    grad_poly[n_modes-1] = n_modes*b
    grad_poly[2*n_modes-1] = 2*n_modes*a
    grad_poly.reverse()
    roots = np.roots(grad_poly)
    current_best = np.Inf
    best_x = 0
    for sol in roots:
        if sol.imag<1e-16:
            sol = sol.real
            if sol>0:
                val = np.polyval(poly,sol)
                if val<current_best:
                    current_best = val
                    best_x = sol
    if current_best==np.Inf:
        print("No solution to scaling !!!")
        return tensor, None

    # We have the optimal scale
    for i in range(n_modes):
        factors[i] *= best_x

    if format_tensor=="tucker":
        return tl.tucker_tensor.TuckerTensor((factors[-1], factors[:-1])), best_x
    return tl.cp_tensor.CPTensor((None, factors)), best_x

if __name__== "__main__":

    # matrix test
    #test_matrix = tl.tensor([[1,3],[2,1]])
    #marginals_matrix = [tl.tensor([0.5, 0.9]),tl.tensor([0.7,0.7])] # feasible
    #test_matrix_scaled, test_scales_matrix, test_marginals_obtained_matrix = tensor_sinkhorn(test_matrix, marginals_matrix)

    #test_tensor = tl.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
    #dims = [10,9,8]
    #test_tensor = 10*tl.abs(tl.randn(dims))
    #random_tensor = tl.abs(tl.randn(dims))
    #marginals = []
    #for mode in range(3):
        #marginals.append( tl.sum( test_tensor*random_tensor, axis=tuple([i for i in range(3) if i!=mode])) ) # feasible
        ##marginals.append(tl.abs(tl.randn([dims[mode]]))) # infeasible?
    #test_tensor_scaled, test_scales, test_marginals_obtained = tensor_sinkhorn(test_tensor, marginals)

    # Test adaptive sinkhorn
    dims = [100,90,80]
    dims_core = [3,3,3]
    itermax=20
    test_tensor = 10*tl.abs(tl.randn(dims_core))
    test_tensor[0,:,:]=0
    test_regs = [tl.abs(tl.randn([dim])) for dim in dims_core]
    test_regs[1][2]=0
    out = tensor_online_sinkhorn(test_tensor,test_regs, 1, hom_g=1, itermax=itermax,verbose=True)
    marginals_computed = out[2]
    # todo assert 0s, some value

    # Test adaptive sinkhorn 2, with diagonal core
    dims = [10,10]
    dims_core = [3,3]
    core = tl.eye(dims_core[0])
    factors = [tl.randn([dims[0],dims_core[0]]),tl.randn([dims[1],dims_core[1]])]
    regs = [tl.sum(tl.abs(fac), axis=0) for fac in factors]
    out = tucker_implicit_sinkhorn_balancing(factors, core, regs, 1, [1, 1, 1], itermax=10)
    regs_after = [tl.sum(tl.abs(fac), axis=0) for fac in factors]
    # two values in regs_after should match (CPD case)

    # Test adaptive sinkhorn 3, with random core
    #dims = [10,10]
    #dims_core = [3,3]
    #core = tl.abs(tl.randn(dims_core))
    #factors = [tl.randn([dims[0],dims_core[0]]),tl.randn([dims[1],dims_core[1]])]
    #regs = [tl.sum(tl.abs(fac), axis=0) for fac in factors]
    #out = tucker_implicit_sinkhorn_balancing(factors, core, regs, 1, [1, 1, 1], itermax=10)
    #regs_after = [tl.sum(tl.abs(fac), axis=0) for fac in factors]