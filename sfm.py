import cv2
import numpy as np

KL = np.array([[3979.911, 0, 1244.772], [0, 3979.911, 1019.507], [0, 0, 1]])
KR = np.array([[3979.911, 0, 1369.115], [0, 3979.911, 1019.507], [0, 0, 1]])
baseline = 193.001
f = 3979.911
distortion = np.zeros((4,1),dtype = np.float32)

# Image Acquisition
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

F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 0.5, 0.99)
pts1 = pts1[mask.ravel() == 1]
pts2 = pts2[mask.ravel() == 1]

E = np.matmul(np.matmul(KL.T,F), KL)

K_inv = np.linalg.inv(KL)

first_inliers = cv2.convertPointsToHomogeneous(pts1)
second_inliers = cv2.convertPointsToHomogeneous(pts2)

_, R, t, _ = cv2.recoverPose(E, np.array(pts1), np.array(pts2), KL)

RL, RR, PL, PR, Q, _, _ = cv2.stereoRectify(KL, distortion, KL, distortion, imgL.shape[:2], R, t, alpha = -1)

mapL1, mapL2 = cv2.initUndistortRectifyMap(KL, distortion, RL, KL, imgL.shape[:2], cv2.CV_32FC1)
mapR1, mapR2 = cv2.initUndistortRectifyMap(KL, distortion, RR, KL, imgR.shape[:2], cv2.CV_32FC1)

img_rect_1 = cv2.remap(imgL, mapL1, mapL2, cv2.INTER_LINEAR)
img_rect_2 = cv2.remap(imgR, mapR1, mapR2, cv2.INTER_LINEAR)

cv2.imshow('Rectified Left', img_rect_1)
cv2.imshow('Rectified Right', img_rect_2)

cv2.waitKey(0)
cv2.destroyAllWindows()



