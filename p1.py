# 1/4
import numpy as np
import matplotlib.pyplot as plt

N = 32
def error(N):
    x = np.linspace(0, 2*np.pi, 10000)
    x_N = np.linspace(0, 2*np.pi, N+1)[:-1]
#    f = np.abs(np.sin(x))**3
    f = 1/(1+2*np.sin(x)**2)
#    f_N = np.abs(np.sin(x_N))**3
    f_N = 1/(1+2*np.sin(x_N)**2)
    k = np.arange(-N/2, N/2)
    Fk = np.fft.fftshift(np.fft.fft(f_N))/N
    S_N = np.zeros_like(x, dtype='float')
    for i in range(N):
        S_N = S_N + Fk[i]*np.exp(1j*k[i]*x) 

    err = np.max(f - np.real(S_N))
    return err

N = np.array([4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
e = np.zeros_like(N, dtype='float')
for i in range(len(N)):
    e[i] = error(N[i])

plt.loglog(N, e, 'o-', color='salmon')
#plt.loglog(N, 1/N**3, '-', color='teal', label='slope 3')
plt.grid(which='both')
plt.xlabel('$N$')
plt.ylabel('Error')
plt.title('$1/(1+2sin(x)^{2})$')
plt.legend()
plt.show()
