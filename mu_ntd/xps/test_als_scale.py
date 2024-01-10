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

import matplotlib.pyplot as plt
plt.figure()
plt.subplot(221)
plt.semilogy(err)
plt.subplot(222)
plt.plot(out[1])
plt.plot(out[2])
plt.plot([np.sqrt(y-lamb) for i in range(len(out[1]))], "--m")
plt.subplot(223)
plt.semilogy(out[3][1:])
#plt.semilogy([y for i in range(len(out[3][1:]))],"--k")
plt.semilogy([y-lamb for i in range(len(out[3][1:]))],"--m")

# convergence rate: err[k-1] - err[k]
err_rate = [err[i]-err[i+1] for i in range(len(err)-1)]
err_rate_rel = [(err[i]-err[i+1])/err[i] for i in range(len(err)-1)]
plt.subplot(224)
plt.semilogy(err_rate[2:], "-b")
plt.semilogy([i for i in range(2,len(err_rate))],out[4][2:],"--m")
#plt.semilogy([i for i in range(2,len(err_rate))],out[5][2:],"--k")
#plt.semilogy([lamb**2 for i in range(2,len(err_rate))], "--m")

plt.figure()
diffx = [(out[1][i+1]-np.sqrt(y-lamb))/(out[1][i]-np.sqrt(y-lamb)) for i in range(1,len(out[1])-1)]
plt.semilogy(diffx) #pb facteur 2 ?
print("Practical and theoretical epsx ratio", diffx[-1], 1-4*lamb/y)

#plt.semilogy(err_rate_rel)

plt.show()