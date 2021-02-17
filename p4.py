# 4/4
import numpy as np
import matplotlib.pyplot as plt
import compute_DCT as dct
import scipy as sc

N = 4
j = np.flip(np.arange(0, N+1))
x = np.cos(np.pi*j/N)
print(x)
#f = np.cos(x)
#f = np.piecewise(x, [x < 0, x >= 0], \
#[lambda x: -2*x-1,lambda x: 2*x-1])
f = x**3
#plt.plot(x, f)
k, Fk = dct.compute_DCT(f)

A = np.zeros((N+1, N+1))
np.fill_diagonal(A[1:], 1)
np.fill_diagonal(A[:,1:], -1)
A[0,:] = 0
A[1,0] = 2
#print(A)

nA = A[1:,:-1]
#print(nA)
b = np.zeros(N+1)
b = 2*k*Fk
bn = b[1:]
print(bn)

phi = np.linalg.solve(nA, bn)

fp = sc.fft.idct(phi, 1)*N

plt.plot(fp)
#plt.show()
