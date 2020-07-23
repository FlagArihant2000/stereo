import cv2
import numpy as np
import os

totlist = sorted(os.listdir('/home/arihant/Downloads/templeRing'))
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
while(imgenum < len(imglist) - 1):
	img1 = cv2.imread(imglist[imgenum])
	img2 = cv2.imread(imglist[imgenum + 1])
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
	pts1 = pts1[mask.ravel() ==1]
	pts2 = pts2[mask.ravel() ==1]
	_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
	
	P1 = np.zeros((3,4))
	P1 = np.matmul(K, P1)
	P2 = np.hstack((R, t))
	P2 = np.matmul(K, P2)
	points1 = pts1.reshape(2, -1)
	points2 = pts2.reshape(2, -1)
	R1, R2, P1, P2, Q, a, b = cv2.stereoRectify(K, D, K, D, (640, 480), R, t)
	map1, map2 = cv2.initUndistortRectifyMap(K, D, R1, P1, (640, 480), cv2.CV_16SC2)
	img1rec = cv2.remap(img1, map1, map2, cv2.INTER_CUBIC)

	map3, map4 = cv2.initUndistortRectifyMap(K, D, R2, P2, (640, 480), cv2.CV_16SC2)
	img2rec = cv2.remap(img2, map3, map4, cv2.INTER_CUBIC)
	
	
	cv2.imshow('image1', img1rec)
	cv2.imshow('image2', img2rec)
	imgenum = imgenum + 1
	if cv2.waitKey(1) & 0xff == ord('q'):
		break
		
cv2.destroyAllWindows()


print(K)
