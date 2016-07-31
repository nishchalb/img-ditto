#!/usr/bin/env python

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import argparse

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
    result = v_map(to_im)
    return result 

def transfer_rgb(from_im, to_im, strength=1.0):
    result = to_im.copy()
    for i in [0,1,2]:
        result[:,:,i] = transfer(from_im[:,:,i], to_im[:,:,i])
    return result * strength + to_im*(1-strength)


def transfer_using_colorspace(from_im, to_im, colorspace_f, colorspace_r, strength = 1.0):
    from_im_cvt = cv2.cvtColor(from_im, colorspace_f)
    to_im_cvt = cv2.cvtColor(to_im, colorspace_f)
    result = to_im_cvt.copy()
    for i in [0,1,2]:
        result[:,:,i] = transfer(from_im_cvt[:,:,i], to_im_cvt[:,:,i])
    return cv2.cvtColor(result, colorspace_r)*strength + to_im * (1 - strength)

def transfer_hsv(from_im, to_im, hue=True, saturation=True, value=True, strength = 1.0):
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
    return cv2.cvtColor(result, cv2.COLOR_HSV2BGR)*strength + to_im * (1 - strength)

def main():
    parser = argparse.ArgumentParser(
            description='''Transfer the style from one image to another using 
            histogram distributions''')
    parser.add_argument('from_im', help="The image who's style you want to use")
    parser.add_argument(
            'to', 
            help="The image(s) you want to transfer the style to", 
            nargs='+'
            )
    parser.add_argument(
            '-o', 
            '--output', 
            help="The suffix to append to the output files"
            )
    parser.add_argument(
            '-s', 
            '--strength', 
            help="""The strength of the transfer, should be between 0.0 and 1.0. 
                  Defaults to 0.8""",
            type=float,
            default=0.8
            )
    parser.add_argument(
            '-c', 
            '--colorspace', 
            help="""The colorspace to use for the transformation. Currently
                    supported are rgb, hsv, and lab. Defaults to lab.""",
            default='lab'
            )

    args = parser.parse_args()
    input_split = args.from_im.split('/')
    input_name, _,input_ext = input_split[-1].partition('.') 
    from_im = cv2.imread(args.from_im)
    if args.output is None:
        suffix = "_styled_like_{}".format(input_name)
    else:
        suffix = args.output

    for img in args.to:
        print "Working on {}".format(img)
        to_im = cv2.imread(img)
        if args.colorspace == 'rgb':
            result = transfer_rgb(from_im, to_im, strength = args.strength)
        elif args.colorspace == 'lab':
            result = transfer_using_colorspace(from_im, to_im, 
                    cv2.COLOR_BGR2LAB, cv2.COLOR_LAB2BGR, 
                    strength = args.strength)
        else:
            result = transfer_hsv(from_im, to_im, strength = args.strength)
        out_split = img.split('/')
        out_name, _, out_ext = out_split[-1].partition('.')
        print '{}{}.{}'.format(out_name, suffix, out_ext)
        cv2.imwrite('{}{}.{}'.format(out_name, suffix, out_ext), result)

if __name__ == "__main__":
    main()
