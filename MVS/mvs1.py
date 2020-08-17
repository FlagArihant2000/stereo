# Multiview Stereo of an object (Using ICP Registration for merging point clouds from disparity maps)
# Dataset: Temple Dataset

# Author: Arihant Gaur


import cv2
import numpy as np
import os

def get_parameters(cam_par):
	K = np.array([[float(cam_par[1]), float(cam_par[2]), float(cam_par[3])], [float(cam_par[4]), float(cam_par[5]), float(cam_par[6])], [float(cam_par[7]), float(cam_par[8]), float(cam_par[9])]])
	R = np.array([[float(cam_par[10]), float(cam_par[11]), float(cam_par[12])], [float(cam_par[13]), float(cam_par[14]), float(cam_par[15])], [float(cam_par[16]), float(cam_par[17]), float(cam_par[18])]])
	t = np.array([[float(cam_par[19])], [float(cam_par[20])], [float(cam_par[20][:-1])]])
	return K, R, t
	
	
def Reprojection3D(image, disparity, f, b):
	Q = np.array([[1, 0, 0, -2964/2], [0, 1, 0, -2000/2],[0, 0, 0, f],[0, 0, -1/b, -124.343/b]])

	points = cv2.reprojectImageTo3D(disparity, Q)
	mask = disparity > disparity.min()
	colors = image

	
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

img_directory = 'templeSparseRing/'

totlist = sorted(os.listdir(img_directory))
imglist = []
for i in totlist:
	if '.png' in i and 'R' in i:
		imglist = imglist + [i]

f = open(img_directory + 'templeR_par.txt')

i = 0
line = f.readline()
line1 = f.readline()

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
R_ref = np.eye(3)
t_ref = np.array([[0],[0],[0]], dtype = np.float32)
D = np.zeros((5,1), dtype = np.float32)
useGT = True

max_disparity = 119
min_disparity = 23
num_disparities = max_disparity - min_disparity
window_size = 5
stereo = cv2.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities, blockSize = 5, uniquenessRatio = 15, speckleWindowSize = 20, speckleRange = 15, disp12MaxDiff = 2, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2)

stereo2 = cv2.ximgproc.createRightMatcher(stereo)

lamb = 8000
sig = 1.5
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(lamb)
wls_filter.setSigmaColor(sig)

while(i < len(imglist) - 1):
	img0 = cv2.imread(img_directory + imglist[i])
	img1 = cv2.imread(img_directory + imglist[i + 1])
	line2 = f.readline()
	cam_par1 = line1.split(' ')
	cam_par2 = line2.split(' ')
	K1, R1, t1 = get_parameters(cam_par1)
	if i == 0:
		R_ref = R1
		t_ref = t1
	K2, R2, t2 = get_parameters(cam_par2)
	
	#print(t1[0][0],t1[1][0],t1[2][0])
	#print(t2[0][0], t2[1][0], t2[2][0])
	c1 = -1 * np.matmul(np.linalg.inv(R1), t1)
	c2 = -1 * np.matmul(np.linalg.inv(R2), t2)
	
	baseline = np.linalg.norm(c1 - c2)
	#print(baseline, np.linalg.norm(t1 - t2))
	img0gray = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	
	kp0, des0 = sift.detectAndCompute(img0gray, None)
	kp1, des1 = sift.detectAndCompute(img1gray, None)
	
	matches = bf.knnMatch(des0, des1, k = 2)
	good = []
	for m,n in matches:
		if m.distance < 0.70 * n.distance:
			good.append(m)
			
	pts0 = np.float32([kp0[m.queryIdx].pt for m in good]).reshape(-1,1,2)
	pts1 = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
	
	E, mask = cv2.findEssentialMat(pts0, pts1, K1, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
	#print(np.linalg.matrix_rank(E))
	
	#R_ref = R_ref.dot(R)
	#t_ref = 
	
	pts0 = pts0[mask.ravel() == 1]
	pts1 = pts1[mask.ravel() == 1]
	
	_, R, t, mask = cv2.recoverPose(E, pts0, pts1, K1)
	#print(R, t)
	
	#P1 = np.hstack((R1, t1))
	#P1 = np.matmul(K1, P1)
	
	#P2 = np.hstack((R2, t2))
	#P2 = np.matmul(K2, P2)
	
	points1 = pts0.reshape(2, -1)
	points2 = pts1.reshape(2, -1)

	R1, R2, P1, P2, Q, a, b = cv2.stereoRectify(K1, D, K2, D, (640, 480), R, t)
	map1, map2 = cv2.initUndistortRectifyMap(K1, D, R1, P1, (640, 480), cv2.CV_16SC2)
	img0rec = cv2.remap(img0, map1, map2, cv2.INTER_CUBIC)

	map3, map4 = cv2.initUndistortRectifyMap(K2, D, R2, P2, (640, 480), cv2.CV_16SC2)
	img1rec = cv2.remap(img1, map3, map4, cv2.INTER_CUBIC)
	
	disparity1 = stereo.compute(img0rec, img1rec)
	disparity1 = np.int16(disparity1)
	
	disparity2 = stereo.compute(img1rec, img0rec)
	disparity2 = np.int16(disparity2)
	
	filteredImg = wls_filter.filter(disparity1, img0, None, disparity2)
	_, filteredImg = cv2.threshold(filteredImg, 0, max_disparity * 16, cv2.THRESH_TOZERO)
	filteredImg = (filteredImg / 16).astype(np.uint8)
	

	
	cv2.imshow('disparity', filteredImg)
	#cv2.imshow('image0', img0rec)
	#cv2.imshow('image1', img1rec)
	
	i = i + 1
	line1 = line2
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
		
cv2.destroyAllWindows()
