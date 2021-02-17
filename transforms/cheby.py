import numpy as np

#=============================================================================#
# Discrete Chebyshev Transform (Forward and Reverse) from Moin P. pp. 190     #
#=============================================================================#

def cheby(xj, f):
    """
    Computes the 1D discrete Chebyshev transform of f
    Parameters:
        xj (numpy array) : (unequal) grid points
        f  (lambda func) : function
    Returns:
        Fk (numpy array) : Chebyshev coefficients

    """
    N = int(len(xj))-1
    Fk = np.zeros_like(xj, dtype='float')
    t = np.arange(0, N+1)*np.pi/N # uniform grid in theta

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

def icheby(t, Fk):
    """
    Computes the 1D discrete inverse Chebyshev transform of f
    Parameters:
        t  (numpy array) : (equal) grid points
        Fk (numpy array) : Chebyshev coefficients
    Returns:
        fc (numpy array) : reconstructed function 

    """
    fc = np.zeros_like(t, dtype='float')
    N = int(len(t))-1

    for k in range(N+1):
        fc = fc + Fk[k]*np.cos(k*t)
    return fc
