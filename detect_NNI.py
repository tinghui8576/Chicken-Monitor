import argparse
import time
from pathlib import Path
import csv
import cv2
import torch
import os
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.quantify import cal_NNI

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

def calibration(pts, img, iorp=True):
    DIM = np.load('cal_parameters_L/DIM.npy')
    k = np.load('cal_parameters_L/K.npy')
    d= np.load('cal_parameters_L/D.npy')
    
    # Just by scaling the matrix coefficients!
    nk = k.copy()
    nk[0,0]=k[0,0]/2
    nk[1,1]=k[1,1]/2
    DIM = tuple(DIM)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(k, d, np.eye(3), nk, DIM, cv2.CV_32FC1)  # Pass k in 1st parameter, nk in 4th parameter
    
    # Calibrate points
    if iorp == True:
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
        dst = torch.from_numpy(dst)
        
        return dst
    # Calibrate img
    else:
        nemImg = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return nemImg

   
    
def detect(save_img=False):
    start = time.time()
    nni = 0
    movement = 0
    moves = []
    record_time = 5
    width1 = 1920
    height1 = 1080
    
    # Load the model 
    # ==========================================================================================================================================================
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print('log')
    
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    # ==========================================================================================================================================================
    
    
    # Load the first image for optical flow
    # ==========================================================================================================================================================
    path_list = os.listdir(source)
    img_file = os.path.join(source, path_list[0])
    
    prev_frame = cv2.imread(img_file)
    prev_frame = calibration([],prev_frame,False)
    prev_frame = cv2.resize(prev_frame, (480,270))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    timestamp_str = os.path.splitext(os.path.basename(img_file))[0]
    ph, pm, _ = timestamp_str.split('-')
    os.remove(img_file)
    print(path_list[0])
    # ==========================================================================================================================================================
    
    
    while True:
        if int(time.strftime("%H")) >= 18:
          break
        # Set Dataloader
        vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
        for path, img, im0s, vid_cap in dataset:
            
            frameIndex = 0
            t0 = time.time()
            timestamp_str = os.path.splitext(os.path.basename(path))[0]
            print(timestamp_str)
            h, m, _ = timestamp_str.split('-')
            
            if img is None:
                time.sleep(15)
                break
                
            curr_frame = calibration([],im0s,False)
            curr_frame = cv2.resize(curr_frame, (480,270))
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
            
            move = calculate_optical_flow(prev_gray, curr_gray)
            os.remove(path)
            if pm != m:
                #  Detect and calculate NNI
                # ===================================================================================================================================================   
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
        
                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]
        
                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]
                t2 = time_synchronized()
        
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t3 = time_synchronized()
    
                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        
                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                    if len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                        det[:, :4]  = calibration(det[:, :4], [], True)
                        
                        dets = []
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                           dets.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf)]) 
        
                        dets = np.asarray(dets)
                        #print(len(dets))
                        if len(dets) > 2:
                          nni = cal_NNI(dets[:, :2], width1, height1)
                          
                          
                    # Print time (inference + NMS)
                    print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                # ===================================================================================================================================================    
                if len(moves) != 0: 
                    movement = sum(moves)/len(moves)
                    print("avg", sum(moves)/len(moves))
                #Time.append(ph+"-"+pm)
                print(ph+"-"+pm)
                print(nni, movement)
                with open("result.csv", 'a') as f:
                    csv_writer = csv.writer(f, delimiter=',') 
                    csv_writer.writerow([str(ph+"_"+pm),movement,nni])
                pm = m
                ph = h
                moves = []
    
            # Updates previous frame
            prev_gray = curr_gray
            moves.append(move)
            #time.append(timestamp_str)
            #print("one", move)
            print(f'Done. ({time.time() - t0:.3f}s)')
            time.sleep(record_time)            
    
    
        
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')
    end = time.time()
    print(end -start)
    #return avg_mov, nni, std_mov, std_nni, frameIndex

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='runs/train/yolo_chicken_det20/weights/best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='./Pictures', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    # print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))
    
    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()


    
