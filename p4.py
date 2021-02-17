# 4/4
import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby

N = np.array([4, 8, 16, 32])

def plot_derivative(N, func=1, method='chebyshev'):
    t = np.arange(0, N+1)*np.pi/N
    x = np.cos(t)
    if func == 1:
        f = lambda x: x**6                               # test function
        dfdx = lambda x: 6*x**5
    elif func == 2:
        f = lambda x: x*np.exp(-(x**2)/2)                # part (a)
        dfdx = lambda x: np.exp(-(x**2)/2)*(1 - x**2)
    else:
        f = lambda x: np.piecewise(x, [x < 0, x >= 0], \
        [lambda x: -2*x-1,lambda x: 2*x-1])              # part (b)
        dfdx = lambda x: np.piecewise(x, [x < 0, x>= 0], [-2,2])

    if method=='chebyshev':
        # compute chebyshev transform
        Fk = cheby.cheby(x, f)
        k = np.arange(0, N+1)
        # assemble bi-diagonal matrix
        A = np.zeros((N+1, N+1))
        np.fill_diagonal(A[1:], 1)
        np.fill_diagonal(A[:,1:], -1)
        A[0,:] = 0
        A[1,0] = 2
        nA = A[1:,:-1]
        # assmble RHS
        b = np.zeros(N+1)
        b = 2*k*Fk
        bn = b[1:]
        # solve bi-diagonal system
        phi = np.linalg.solve(nA, bn)
        # set last coefficient to 0
        phi = np.append(phi, 0.0)
        # inverse transform
        fp = cheby.icheby(t, phi)
        plt.plot(x, fp, '-o', label='N = {}'.format(N))
    if method=='finiteDiff':
        # compute derivative...

        return dfdx

for i in range(len(N)):
    dfdx = plot_derivative(N[i], 1, 'chebyshev')

# exact derivative
x = np.linspace(-1, 1, 128)
plt.plot(x, dfdx(x), '.', label='exact')
plt.xlabel('$x$')
plt.ylabel('$df$/$dx$')
plt.title('$f(x) = 1$')
plt.legend()
plt.show()

