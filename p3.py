# 3/4
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import compute_DCT as dct
import dct

N = np.array([4, 8, 16, 32])

def chebyshev(N):
    # cosine mesh
    j = np.flip(np.arange(0, N+1))
    x = np.cos(np.pi*j/N)
    # function evaluated on cosine mesh (uncomment one)
    #f = lambda x: x*np.exp(-(x**2)/2)
    f = lambda x: x**3
    #f = np.piecewise(x, [x < 0, x >= 0], \
    #[lambda x: -2*x-1,lambda x: 2*x-1])
    # compute Chebyshev coefficients
    #k, Fk = dct.compute_DCT(f)
    Fk = dct.DCT(x, f)
    # plot coefficients
    k = np.arange(0, N+1)
    plt.plot(k, np.abs(Fk),'-o', label='N = {}'.format(N))

for i in range(len(N)):
    chebyshev(N[i])

plt.xlabel('$n$')
plt.ylabel('$a_{n}$')
plt.title('Piecewise function')
#plt.xlim([-0.5,8.5])
plt.legend()
plt.show()
