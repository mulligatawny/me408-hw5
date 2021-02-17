# 3/4
import numpy as np
import matplotlib.pyplot as plt
from transforms import cheby

N = np.array([4, 8, 16, 32])

def plot_transform(N, func):
    t = np.arange(0, N+1)*np.pi/N # equal grid
    x = np.cos(t) # unequal grid
    if func == 1:
        f = lambda x: x**6 # test function
    elif func == 2:
        f = lambda x: x*np.exp(-(x**2)/2) # part (a)
    else:
        f = lambda x: np.piecewise(x, [x < 0, x >= 0], \
        [lambda x: -2*x-1,lambda x: 2*x-1]) # part (b)
    Fk = cheby.cheby(x, f)
    k = np.arange(0, N+1)
    plt.plot(k, np.abs(Fk), '-o', label='N = {}'.format(N))

for i in range(len(N)):
    plot_transform(N[i], 1)

plt.xlabel('$n$')
plt.ylabel('$|a_{n}|$')
plt.legend()
plt.show()
