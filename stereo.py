import cv2
import numpy as np
import random

cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)

imgL = cv2.imread("/home/arihant/Desktop/sfm/Structure-From-Motion-master/data/Stereo/im0.png")
imgR = cv2.imread("/home/arihant/Desktop/sfm/Structure-From-Motion-master/data/Stereo/im1.png")

imgL = cv2.pyrDown(imgL)
imgR = cv2.pyrDown(imgR)

f = 3979.911
baseline = 193.001


window_size = 3
min_disp = 1
num_disp = 16

stereo = cv2.StereoSGBM_create(minDisparity = min_disp, numDisparities = num_disp, blockSize = 16, P1 = 8 * 3 * window_size**2, P2 = 32 * 3 * window_size**2, disp12MaxDiff = 1, uniquenessRatio = 10, speckleWindowSize = 100, speckleRange = 32)

disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

cv2.imshow('disparity',disp)
cv2.imshow('Left Image', imgL)
cv2.imshow('Right Image', imgR)
cv2.waitKey(0)
cv2.destroyAllWindows()
