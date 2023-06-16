import numpy as np
import cv2

def calibrate_map():
    DIM = np.load('cal_parameters_L/DIM.npy')
    k = np.load('cal_parameters_L/K.npy')
    d= np.load('cal_parameters_L/D.npy')  
    nk = k.copy()
    nk[0,0]=k[0,0]/2
    nk[1,1]=k[1,1]/2
    # Just by scaling the matrix coefficients!
    DIM = tuple(DIM)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), nk, DIM, cv2.CV_32FC1) 
   
    return map1, map2


def calibrate_img(img, map1, map2):
    nemImg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
   
    return nemImg

def calibrate_pts(pts, map1, map2):
    
    # Calibrate points
    idloc = []
    idloc2 = []
    for pt in pts:
        pt1 = []
        pt2 = []
        pt1.append(int(pt[0]))
        pt1.append(int(pt[1]))
        idloc.append(pt1)
        pt2.append(int(pt[2]))
        pt2.append(int(pt[3]))
        idloc2.append(pt2)
    idloc = np.asarray(idloc, dtype=float)
    idloc2 = np.asarray(idloc2, dtype=float)
    idloc = np.expand_dims(idloc, 1)   
    idloc2 = np.expand_dims(idloc2, 1)
      
    dst = cv2.fisheye.undistortPoints(idloc, k, d, cv2.CV_16SC2, P = nk)  # Pass k in 1st parameter, nk in 4th parameter
    idloc2 = cv2.fisheye.undistortPoints(idloc2, k, d, cv2.CV_16SC2, P = nk)  # Pass k in 1st parameter, nk in 4th parameter  
    dst = dst.squeeze()
    idloc2 = idloc2.squeeze()
    dst = np.hstack((dst, idloc2))  
        
    return dst
    