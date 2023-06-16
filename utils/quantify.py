import numpy as np
import math
import cv2
from sklearn.neighbors import NearestNeighbors

def cal_NNI(det, width1, height1):
    #print(width1, height1)
    area = width1*height1
    num = len(det)
    nbrs = NearestNeighbors(n_neighbors=2).fit(det)
    dist, index = nbrs.kneighbors(det)
    d_obs = sum(dist)[1]/float(num)
    d_exp = 0.5/math.sqrt(num/float(area))
    NNI = d_obs/d_exp

    return NNI

def cal_optical_flow(prev_gray, next_gray):
    # define parameters for optical flow calculation
    pyr_scale = 0.5
    levels = 1
    winsize = 80
    iterations = 1
    poly_n = 5
    poly_sigma = 1.1
    flags = 0
    flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags)
    # convert optical flow to polar coordinates
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.sum(magnitude)