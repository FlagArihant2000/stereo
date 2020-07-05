# STEREO: RECONSTRUCTING A 3D SCENE FROM STEREO IMAGE PAIRS

# AUTHOR: ARIHANT GAUR
# ORGANIZATION: IvLabs, VNIT


import cv2
import numpy as np
import random

def DrawEpipolarLines(img1, img2, lines, pts1, pts2):
	r, c = img1.shape[0], img1.shape[1]
	for r, pt1, pt2 in zip(lines, pts1, pts2):
		color = tuple(np.random.randint(0, 255, 3).tolist())
		x0, y0 = map(int, [0, -r[2] / r[1] ])
		x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1] ])
		img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
		img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
		img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)

	return img1, img2
	
def DisplayEpipolarLines(img1, img2, pts1, pts2):
	linesLeft = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, F)
	linesLeft = linesLeft.reshape(-1,3)
	img5, img6 = DrawEpipolarLines(img1, img2, linesLeft, pts1, pts2)

	linesRight = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, F)
	linesRight = linesLeft.reshape(-1,3)
	img3, img4 = DrawEpipolarLines(img2, img1, linesRight, pts2, pts1)
	
	return img5, img3

def ImageRectification(image1, image2, pts1, pts2, F, KL, KR, d):
	height, width = image1.shape[0], image1.shape[1]
	image_size = (width, height)
	retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, image_size)
	# Perform rectify shearing, after the whole algorithm is done
	
	K_invL = np.linalg.inv(KL)
	K_invR = np.linalg.inv(KR)
	R1 = np.matmul(np.matmul(K_invL,H1),KL)
	R2 = np.matmul(np.matmul(K_invR,H2),KR)
	
	mapx1, mapy1 = cv2.initUndistortRectifyMap(KL, d, R1, KL, image_size, cv2.CV_16SC2)
	mapx2, mapy2 = cv2.initUndistortRectifyMap(KR, d, R2, KR, image_size, cv2.CV_16SC2)
	
	
	rectified1 = cv2.remap(image1, mapx1, mapy1, interpolation = cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT)
	rectified2 = cv2.remap(image2, mapx2, mapy2, interpolation = cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT)
	
	return rectified1, rectified2
	
def Reprojection3D(image, disparity, f, b):
	#Q = np.array([[1, 0, 0, -2964/2], [0, 1, 0, -2000/2],[0, 0, 0, f],[0, 0, -1/b, 124.343/b]])
	Q = np.array([[1, 0, 0, -2964/2], [0, 1, 0, 2000/2],[0, 0, 0, f],[0, 0, 0, 1]])
	#Q = np.array([[1, 0, 0, -1244.772], [0, 1, 0, -1019.507],[0, 0, 0, f],[0, 0, -1/b, 124.343/b]])
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
	
# Naming Output Windows
cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)
cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)

# Dataset parameters
KL = np.array([[3979.911, 0, 1244.772], [0, 3979.911, 1019.507], [0, 0, 1]])
KR = np.array([[3979.911, 0, 1369.115], [0, 3979.911, 1019.507], [0, 0, 1]])
D = np.zeros((5,1), dtype = np.float32)
baseline = 193.001
f = 3979.911

# Image Acquisition
imgL = cv2.pyrDown(cv2.imread('/home/arihant/stereo/im0.png'))
imgR = cv2.pyrDown(cv2.imread('/home/arihant/stereo/im1.png'))

imgLgray = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgRgray = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# Feature Extraction using SIFT
sift = cv2.xfeatures2d.SIFT_create()

kp1, des1 = sift.detectAndCompute(imgLgray, None)
kp2, des2 = sift.detectAndCompute(imgRgray, None)

# Feature Matching using Brute Force KNN Matching and Outlier Rejection 
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k = 2)

good = []
pts1 = []
pts2 = []
for m,n in matches:
	if m.distance < 0.80 * n.distance:
		good.append(m)
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)

# Fundamental Matrix Calculation
F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

# Checking Epipolar Lines. Uncomment for verification
#imgL, imgR = DisplayEpipolarLines(imgL, imgR, pts1, pts2)

# Stereo Rectification
imgLrec, imgRrec = ImageRectification(imgL, imgR, pts1, pts2, F, KL, KR, D)
#imgL, imgR = DisplayEpipolarLines(imgL, imgR, pts1, pts2)

# Disparity Map
max_disparity = 128
min_disparity = 0
num_disparities = max_disparity - min_disparity
window_size = 3
stereo = cv2.StereoSGBM_create(min_disparity, num_disparities, window_size)
disparity = stereo.compute(imgLrec, imgRrec)
#cv2.filterSpeckles(disparity, 0, 400, max_disparity - 5)
_, disparity = cv2.threshold(disparity, 0, max_disparity * 16, cv2.THRESH_TOZERO)
disparity = (disparity / 16).astype(np.uint8)

# 3D Reprojection
img3D = Reprojection3D(imgL, disparity, f, baseline)

cv2.imshow('Left Image',imgL)
cv2.imshow('Right Image', imgR)
cv2.imshow('disparity', disparity)
cv2.waitKey(0)
cv2.destroyAllWindows()
