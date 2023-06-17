import numpy as np
import argparse
import cv2
import csv
import time
import os
import glob
import math
#from pathlib import Path



def calculate_optical_flow(prev_gray, next_gray):
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


def calibration(img):
    DIM = np.load('cal_parameters_L/DIM.npy')
    k = np.load('cal_parameters_L/K.npy')
    d= np.load('cal_parameters_L/D.npy')  
    nk = k.copy()
    nk[0,0]=k[0,0]/2
    nk[1,1]=k[1,1]/2
    # Just by scaling the matrix coefficients!
    DIM = tuple(DIM)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), nk, DIM, cv2.CV_32FC1) 
    nemImg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
   
    return nemImg
    
def process_videos():
    record_time = 5
    source = opt.source
    output_dir = './'
    # define video properties
    frame_width = 1920
    frame_height = 1080
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # initialize variables for chicken velocity
    num_pixels = frame_width * frame_height
    mean_vx, mean_vy = 0, 0
    move = 0
    count = 0
    start = False
    movement = []
    t = []
    # loop over videos in the 'L' folder
    path_list = os.listdir(source)
    path_list.sort(key=lambda x:os.path.getctime(os.path.join(source, x)))
    img_file = os.path.join(source, path_list[0])
    
    prev_frame = cv2.imread(img_file)
    prev_frame = calibration(prev_frame)
    prev_frame = cv2.resize(prev_frame, (480,270))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    os.remove(img_file)
    print(path_list[0])
    
    while True:
        if int(time.strftime("%H")) >= 20:
          break
        path_list = os.listdir(source)
        path_list.sort(key=lambda x:os.path.getctime(os.path.join(source, x)))
        
        for filename in path_list:
            t0 = time.time()
            if filename.endswith('.jpg'):
                
                img_file = os.path.join(source, filename)
                print(img_file)
                # get timestamp from video file name
                timestamp_str = os.path.splitext(os.path.basename(img_file))[0]
                h, m, _ = timestamp_str.split('-')
                print(h,m)
                # ret, frame = cap.read()
                curr_frame = cv2.imread(img_file)
                if curr_frame is None:
                  time.sleep(15)
                  break
                curr_frame = calibration(curr_frame)
                curr_frame = cv2.resize(curr_frame, (480,270))
                curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

                move = calculate_optical_flow(prev_gray, curr_gray)

                # Updates previous frame
                prev_gray = curr_gray
                #movement.append(move)
                #time.append(timestamp_str)
                print(move)
                print(f'Done. ({time.time() - t0:.3f}s)')
                time.sleep(record_time)
                os.remove(img_file)
            with open(os.path.join(output_dir, 'velocity_result.csv'), 'a') as f:
                csv_writer = csv.writer(f, delimiter=',') 
                csv_writer.writerow([str(timestamp_str),move])    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='./0', help='source')  # file/folder, 0 for webcam
    opt = parser.parse_args()
    process_videos()
    
              
