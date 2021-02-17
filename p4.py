# 4/4
import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby

N = 32
t = np.arange(0, N+1)*np.pi/N
x = np.cos(t)
func = 3
if func == 1:
    f = lambda x: x**6 # test function
elif func == 2:
    f = lambda x: x*np.exp(-(x**2)/2) # part (a)
else:
    f = lambda x: np.piecewise(x, [x < 0, x >= 0], \
    [lambda x: -2*x-1,lambda x: 2*x-1]) # part (b)
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

phi = np.linalg.solve(nA, bn)

t2 = np.arange(0, N)*np.pi/N
x2 = np.cos(t2)

# inverse transform
fp = cheby.icheby(t2, phi)

plt.plot(x2, fp)
plt.show()

