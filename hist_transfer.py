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
        scipy.interpolate.interp1d: A linearly interpolated estimator of the cdf.
    """
    bins = np.arange(257) #include right edge in bins
    y = np.histogram(X, bins, density=True)
    cdf_hist = np.cumsum(y[0])
    x_range = y[1][:256]
    P = interp1d(x_range, cdf_hist)
    return P

def transfer(from_im, to_im):
    """ Transfers histogram from from_im to to_im

    Args:
        from_im (np.ndarray): A 2-dimensional numpy array
        to_im (np.ndarray): A 2-dimensional numpy array

    Returns:
        np.ndarray: The new values for to_im
    """
    F = est_cdf(to_im)
    G = est_cdf(from_im)
    G_inv = np.interp(F.y, G.y, G.x, left=0.0, right=1.0)
    mapping = {}
    x_range = np.arange(256)
    for n, i in enumerate(x_range):
        val = F(i)
        xj = G_inv[n]
        xj = round(xj)
        mapping[i] = xj
    v_map = np.vectorize(lambda x: mapping[x])
    return v_map(to_im)
