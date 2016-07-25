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

def transfer_rgb(from_im, to_im):
    result = to_im.copy()
    for i in [0,1,2]:
        result[:,:,i] = transfer(from_im[:,:,i], to_im[:,:,i])
    return result


def transfer_using_colorspace(from_im, to_im, colorspace_f, colorspace_r):
    from_im_cvt = cv2.cvtColor(from_im, colorspace_f)
    to_im_cvt = cv2.cvtColor(to_im, colorspace_f)
    result = to_im_cvt.copy()
    for i in [0,1,2]:
        result[:,:,i] = transfer(from_im_cvt[:,:,i], to_im_cvt[:,:,i])
    return cv2.cvtColor(result, colorspace_r)

def transfer_hsv(from_im, to_im, hue=True, saturation=True, value=True):
    from_im_hsv = cv2.cvtColor(from_im, cv2.COLOR_BGR2HSV)
    to_im_hsv = cv2.cvtColor(to_im, cv2.COLOR_BGR2HSV)
    result = to_im_hsv.copy()
    channels = []
    if hue:
        channels.append(0)
    if saturation:
        channels.append(1)
    if value:
        channels.append(2)
    for i in channels:
        result[:,:,i] = transfer(from_im_hsv[:,:,i], to_im_hsv[:,:,i])
    return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
