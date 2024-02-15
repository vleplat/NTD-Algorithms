import numpy as np

y = 10
#lamb=2*1e-1 # LINEAR CONVERGENCE
lamb=1e-3 # almost sublinear convergence

def als(y,x0=np.sqrt(y)-0.1,z0=np.sqrt(y)+0.1,lamb=1e-3, itermax=20000):
    x = x0
    z = z0
    xs = [x]
    zs = [z]
    err=[(y-x*z)**2+lamb*(x**2+z**2)]
    fbound=[]
    fbound2=[]
    for k in range(itermax):
        # x update
        x_old = x
        z_old = z
        x = z*y/(z**2+lamb)
        dk = x_old*z-x*z
        e1 = x_old - np.sqrt(y-lamb)
        e2 = z - np.sqrt(y-lamb)
        fbound2.append(dk**2) # bound with lamb neglected
        fbound.append(16*lamb**2*e2**2/y) # bound with lamb neglected
        err.append((y-x*z)**2+lamb*(x**2+z**2))

        # z update
        z = x*y/(x**2+lamb)
        dk = x*z_old-x*z
        err.append((y-x*z)**2+lamb*(x**2+z**2))
        fbound2.append(dk**2)
        e1 = x - np.sqrt(y-lamb)
        e2 = z - np.sqrt(y-lamb)
        fbound.append(16*lamb**2*e1**2/y) # bound with lamb neglected
        xs.append(x)
        zs.append(z)
    return err, xs, zs, [xs[i]*zs[i] for i in range(len(xs))], fbound, fbound2

out = als(y,lamb=lamb)
err = out[0]

import matplotlib
import matplotlib.pyplot as plt

font = {'size'   : 16}
matplotlib.rc('font', **font)
matplotlib.rcParams['figure.figsize'] = [16.0, 12.0]
matplotlib.rcParams['lines.linewidth'] = 2.0

plt.figure()
plt.subplot(221)
plt.title('Cost function value with respect to iterations')
plt.semilogy([0.5*i for i in range(len(err))], err)
plt.locator_params(axis='x',nbins=4)
plt.subplot(222)
plt.plot(out[1])
plt.plot(out[2])
plt.title('Values of $x_1$, $x_2$ and $\\sqrt{y-\\lambda}$')
plt.plot([np.sqrt(y-lamb) for i in range(len(out[1]))], "--m")
plt.legend(['$x_1$', '$x_2$', '$\\sqrt{y-\\lambda}$'])
plt.locator_params(axis='x',nbins=4)
plt.locator_params(axis='y',nbins=4)
plt.subplot(223)
plt.title('Values of $x_1x_2$ and $y-\\lambda$')
plt.xlabel('Iteration')
plt.semilogy(out[3][1:])
plt.semilogy([y-lamb for i in range(len(out[3][1:]))],"--m")
plt.locator_params(axis='x',nbins=4)
plt.legend(['$x_1x_2$','$y-\\lambda$'])

# convergence rate: err[k-1] - err[k]
err_rate = [err[i]-err[i+1] for i in range(len(err)-1)]
err_rate_rel = [(err[i]-err[i+1])/err[i] for i in range(len(err)-1)]
plt.subplot(224)
plt.title('Error decrease per iteration')
plt.xlabel('Iteration')
plt.semilogy([0.5*i for i in range(2,len(err_rate))],err_rate[2:], "-b")
plt.semilogy([i/2 for i in range(2,len(err_rate))],out[4][2:],"--m")
plt.legend(['Measured', 'Theory'])
plt.locator_params(axis='x',nbins=4)

#plt.figure()
#diffx = [(out[1][i+1]-np.sqrt(y-lamb))/(out[1][i]-np.sqrt(y-lamb)) for i in range(1,len(out[1])-1)]
#plt.semilogy(diffx) #pb facteur 2 ?
#print("Practical and theoretical epsx ratio", diffx[-1], 1-4*lamb/y)

#plt.show()

plt.savefig('Results/ALS_conv.pdf')