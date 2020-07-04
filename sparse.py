import cv2
import numpy as np

def ImageRectification(image1, image2, K, d, pts1, pts2):
	F, mask = cv2.findFundamentalMat(pts1, pts2)
	pts1 = pts1[mask.ravel() == 1]
	pts2 = pts2[mask.ravel() == 1]
	
	(height, width) = image1.shape
	image_size = (width, height)
	retval, H1, H2 = cv2.stereoRectifyUncalibrated(pts1, pts2, F, image_size)
	# Perform rectify shearing, after the whole algorithm is done
	
	K_inv = np.linalg.inv(K)
	R1 = np.matmul(np.matmul(K_inv,H1),K)
	R2 = np.matmul(np.matmul(K_inv,H2),K)
	
	mapx1, mapy1 = cv2.initUndistortRectifyMap(K, d, R1, K, image_size, cv2.CV_16SC2)
	mapx2, mapy2 = cv2.initUndistortRectifyMap(K, d, R2, K, image_size, cv2.CV_16SC2)
	
	palette1 = set(image1.flatten())
	palette2 = set(image2.flatten())
	
	colors = set(range(256))
	
	key1 = colors.difference(palette1).pop()
	key2 = colors.difference(palette2).pop()
	
	rectified1 = cv2.remap(image1, mapx1, mapy1, interpolation = cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue = key1)
	rectified2 = cv2.remap(image2, mapx2, mapy2, interpolation = cv2.INTER_CUBIC, borderMode = cv2.BORDER_CONSTANT, borderValue = key2)
	
	rectified1_mask = np.ndarray(image1.shape, dtype = bool)
	rectified2_mask = np.ndarray(image1.shape, dtype = bool)
	rectified1_mask.fill(True)
	rectified2_mask.fill(True)
	
	rectified1_mask[rectified1 == key1] = False
	rectified2_mask[rectified2 == key2] = False
	
	return rectified1, rectified2, rectified1_mask, rectified2_mask
	

# Intrinsic Camera Parameters
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

# Image Rectification
imgLrec, imgRrec, maskL, maskR = ImageRectification(imgLgray, imgRgray, KL, distortion, pts1, pts2)

cv2.imshow('Left Rectified', imgLrec)
cv2.imshow('Right Rectified', imgRrec)
cv2.waitKey(0)
cv2.destroyAllWindows()


