import cv2
import numpy as np
import os

img_directory = '/home/arihant/Downloads/templeRing/'
totlist = sorted(os.listdir(img_directory))
imglist = []
for x in totlist:
	if '.png' in x or '.jpg' in x or '.jpeg' in x:
		imglist = imglist + [x]
#print(imglist)
imgenum = 0
K = np.array([[1520.400000, 0.000000, 302.320000], [0.000000, 1525.900000, 246.870000], [0.000000, 0.000000, 1.000000]])
D = np.zeros((5,1), dtype = np.float32)

sift = cv2.xfeatures2d.SIFT_create()
bf = cv2.BFMatcher()
R_ref = np.eye(3)
t_ref = np.array([[0],[0],[0]], dtype = np.float32)

max_disparity = 199
min_disparity = 23
num_disparities = max_disparity - min_disparity
window_size = 5
stereo = cv2.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities, blockSize = 5, uniquenessRatio = 5, speckleWindowSize = 5, speckleRange = 5, disp12MaxDiff = 2, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2)


while(imgenum < len(imglist) - 1):
	img1 = cv2.imread(img_directory + imglist[imgenum])
	img2 = cv2.imread(img_directory + imglist[imgenum + 1])
	img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
	img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
	
	kp1, des1 = sift.detectAndCompute(img1gray, None)
	kp2, des2 = sift.detectAndCompute(img2gray, None)
	
	matches = bf.knnMatch(des1, des2, k = 2)
	
	good = []
	for m,n in matches:
		if m.distance < 0.70 * n.distance:
			good.append(m)
			
	pts1 = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
	pts2 = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
	
	E, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
	print(np.linalg.matrix_rank(E))
	if E is not None and np.linalg.matrix_rank(E) == 2:
		pts1 = pts1[mask.ravel() ==1]
		pts2 = pts2[mask.ravel() ==1]
		_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
		

		P1 = np.hstack((R_ref, t_ref))
		P1 = np.matmul(K, P1)
		
		R_ref = R_ref.dot(R)
		t_ref = t_ref + R_ref.dot(t)
		
		P2 = np.hstack((R_ref, t_ref))
		P2 = np.matmul(K, P2)
		
		points1 = pts1.reshape(2, -1)
		points2 = pts2.reshape(2, -1)
		#R1, R2, P1, P2, Q, a, b = cv2.stereoRectify(K, D, K, D, (640, 480), R, t)
		#map1, map2 = cv2.initUndistortRectifyMap(K, D, R1, P1, (640, 480), cv2.CV_16SC2)
		#img1rec = cv2.remap(img1, map1, map2, cv2.INTER_CUBIC)

		#map3, map4 = cv2.initUndistortRectifyMap(K, D, R2, P2, (640, 480), cv2.CV_16SC2)
		#img2rec = cv2.remap(img2, map3, map4, cv2.INTER_CUBIC)
		
		#if imgenum == 0:
		#	disparity1 = stereo.compute(img1rec, img2rec)
		#else:
		#	disparity2 = stereo.compute(img1rec, img2rec)
			
	cv2.imshow('image', img1)
	#cv2.imshow('image1', img1rec)
	#cv2.imshow('image2', img2rec)
	#cv2.imshow('disparity', disparity1)
	#if imgenum != 0:
	#	cv2.imshow('disparity2', disparity2)
	imgenum = imgenum + 1
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
		
cv2.destroyAllWindows()


print(K)
