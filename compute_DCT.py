import numpy as np
import scipy as sc

def compute_DCT(f):
    """Compute the 1D discrete cosine transform 

    Parameter:
        f  (numpy array): function
    Return:
        k  (numpy array): modes
        Fk (numpy array): cosine transform coefficients
    """
    N = len(f)
    Fk = sc.fft.dct(f, 1)/N
#    Fk[0] = Fk[0]/2
#    Fk[-1] = Fk[-1]/2
    k = np.arange(0, N)
    return k, Fk
