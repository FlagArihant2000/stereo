"""
STEREO 3D RECONSTRUCTION

AUTHOR: Arihant Gaur
ORGANIZATION: IvLabs, VNIT
"""

import cv2
import numpy as np


def Reprojection3D(image, disparity, f, b):
	Q = np.array([[1, 0, 0, -2964/2], [0, 1, 0, -2000/2],[0, 0, 0, f],[0, 0, -1/b, -124.343/b]])
	#Q = np.array([[1, 0, 0, -2964/2], [0, 1, 0, 2000/2],[0, 0, 0, f],[0, 0, 0, 1]])
	#Q = np.array([[1, 0, 0, -1244.772], [0, 1, 0, -1019.507],[0, 0, 0, f],[0, 0, -1/b, 124.343/b]])
	#Q = np.array([[ 1.00000000e+00, 0.00000000e+00, 0.00000000e+00, -6.68420120e+02], [ 0.00000000e+00, 1.00000000e+00, 0.00000000e+00, -5.09922611e+02], [ 0.00000000e+00,  0.00000000e+00, 0.00000000e+00, 1.98995544e+03], [ 0.00000000e+00, 0.00000000e+00, 1.00000000e+00, -0.00000000e+00]])

	#Q = np.array([[1, 0, 0, 0], [0, -1, 0, 0],[0, 0, f * 0.05, 0],[0, 0, 0, 1]])
	points = cv2.reprojectImageTo3D(disparity, Q)
	mask = disparity > disparity.min()
	
	colors = image
	#cv2.imshow('colors', colors)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()
	
	out_points = points[mask]
	out_colors = image[mask]
	
	verts = out_points.reshape(-1,3)
	colors = out_colors.reshape(-1,3)
	verts = np.hstack([verts, colors])
	
	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
		'''
	with open('stereo.ply', 'w') as f:
		f.write(ply_header %dict(vert_num = len(verts)))
		np.savetxt(f, verts, '%f %f %f %d %d %d')

cv2.namedWindow('disparity1', cv2.WINDOW_NORMAL)
cv2.namedWindow('disparity2', cv2.WINDOW_NORMAL)
cv2.namedWindow('WLS Filtered Disparity', cv2.WINDOW_NORMAL)
cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)

K = np.array([[3979.911, 0, 1369.115], [0, 3979.911, 1019.507], [0, 0, 1]], dtype = np.float32)
D = np.zeros((5,1), dtype = np.float32)
downscale = 2
K[0,0] = K[0,0] / float(downscale)
K[1,1] = K[1,1] / float(downscale)
K[0,2] = K[0,2] / float(downscale)
K[1,2] = K[1,2] / float(downscale)

#imgL = cv2.imread('/home/arihant/stereo/im0.png')
#imgR = cv2.imread('/home/arihant/stereo/im1.png')
imgL = cv2.pyrDown(cv2.imread('/home/arihant/stereo/im0.png'))
imgR = cv2.pyrDown(cv2.imread('/home/arihant/stereo/im1.png'))


imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Feature Extraction
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(imgLgray, None)
kp2, des2 = sift.detectAndCompute(imgRgray, None)

# Feature Matching and Outlier Rejection
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k = 2)

good = []
for m,n in matches:
	if m.distance < 0.70 * n.distance:
		good.append(m)
		
pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

E, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
pts1 = pts1[mask.ravel() ==1]
pts2 = pts2[mask.ravel() ==1]
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

P1 = np.zeros((3,4))
P1 = np.matmul(K, P1)
P2 = np.hstack((R, t))
P2 = np.matmul(K, P2)
points1 = pts1.reshape(2, -1)
points2 = pts2.reshape(2, -1)
#cloud = cv2.triangulatePoints(P1, P2, pts1, pts2).reshape(-1, 4)[:, :3]
#ret, R, t, inliers = cv2.solvePnPRansac(cloud, pts2, K, D, cv2.SOLVEPNP_ITERATIVE)

R1, R2, P1, P2, Q, a, b = cv2.stereoRectify(K, D, K, D, (1482, 1000), R, t)
map1, map2 = cv2.initUndistortRectifyMap(K, D, R1, P1, (1482, 1000), cv2.CV_16SC2)
imgLrec = cv2.remap(imgL, map1, map2, cv2.INTER_CUBIC)

map3, map4 = cv2.initUndistortRectifyMap(K, D, R2, P2, (1482, 1000), cv2.CV_16SC2)
imgRrec = cv2.remap(imgR, map3, map4, cv2.INTER_CUBIC)

max_disparity = 199
min_disparity = 23
num_disparities = max_disparity - min_disparity
window_size = 5
stereo = cv2.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities, blockSize = 5, uniquenessRatio = 5, speckleWindowSize = 5, speckleRange = 5, disp12MaxDiff = 2, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2)

stereo2 = cv2.ximgproc.createRightMatcher(stereo)

lamb = 8000
sig = 1.5
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(lamb)
wls_filter.setSigmaColor(sig)

disparity = stereo.compute(imgLrec, imgRrec)

#disparity = np.int16(disparity)
#_, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
#disparity = (disparity / 16).astype(np.uint8)

disparity2 = stereo2.compute(imgRrec, imgLrec)
disparity2 = np.int16(disparity2)

#_, disparity2 = cv2.threshold(disparity2, 0, max_disparity * 16, cv2.THRESH_TOZERO)
#disparity2 = (disparity2 / 16).astype(np.uint8)

filteredImg = wls_filter.filter(disparity, imgL, None, disparity2)
_, filteredImg = cv2.threshold(filteredImg, 0, max_disparity * 16, cv2.THRESH_TOZERO)
filteredImg = (filteredImg / 16).astype(np.uint8)


baseline = 193.001/2
f = 3979.911/2

Reprojection3D(imgL, filteredImg, f, baseline)

cv2.imshow('Left Image', imgLrec)
cv2.imshow('Right Image', imgRrec)
cv2.imshow('disparity1', disparity)
cv2.imshow('disparity2', disparity2)
cv2.imshow('WLS Filtered Disparity', filteredImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

