import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def est_cdf(X):
    """ Returns a function representing the cdf of the input image, as well as
            the inverse of that function

    Args:
        X (np.ndarray): A 2-dimensional numpy array representing an image

    Returns:
        (scipy.interpolate.interp1d, scipy.interpolate.interp1d): The first 
            element is a linearly interpolated estimator of the cdf. The second
            is its inverse
    """
    bins = np.arange(257) #include right edge in bins
    y = np.histogram(X, bins, density=True)
    cdf_hist = np.cumsum(y[0])
    x_range = y[1][:256]
    P = interp1d(x_range, cdf_hist)
    P_inv = interp1d(cdf_hist, x_range, bounds_error=False, fill_value=(0.0, 1.0))
    return (P, P_inv)

def transfer(from_im, to_im):
    """ Transfers histogram from from_im to to_im

    Args:
        from_im (np.ndarray): A 2-dimensional numpy array
        to_im (np.ndarray): A 2-dimensional numpy array

    Returns:
        None: Mutates to_im
    """
    F, F_inv = est_cdf(to_im)
    G, G_inv = est_cdf(from_im)
    mapping = {}
    x_range = np.arange(256)
    plt.plot(x_range, F(x_range), 'r--', x_range, G(x_range), 'b--')
    plt.show()
    for i in x_range:
        val = F(i)
        xj = G_inv(val)
        xj = round(xj)
        mapping[i] = xj
    return mapping
