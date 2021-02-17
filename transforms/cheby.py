import numpy as np

#=============================================================================#
#
#=============================================================================#

def cheby(xj, f):


    N = int(len(xj))-1
    Fk = np.zeros_like(xj, dtype='float')
    t = np.arange(0, N+1)*np.pi/N

    for k in range(N+1):
        ak = 0.0
        for j in range(N+1):
            if j == 0 or j == N:
                ak = ak + f(xj[j])/2.0*np.cos(k*t[j])
            else:
                ak = ak + f(xj[j])*np.cos(k*t[j])

        if k == 0 or k == N:
            ak = ak/N
        else:
            ak = ak*2/N

        Fk[k] = ak
    return Fk

def icheby(xj, Fk):
    """
    Computes the 1D discrete inverse cosine transform of f
    Parameters:
        xj (numpy array) : grid points
        Fk (numpy array) : DCT coefficients
    Returns:
        fc (numpy array) : reconstructed function 

    """
    fc = np.zeros_like(xj, dtype='float')
    N = int(len(xj))-1

    for k in range(N+1):
        fc = fc + Fk[k]*np.cos(k*xj)

    return fc
