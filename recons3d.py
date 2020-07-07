# STEREO 3D RECONSTRUCTION

# AUTHOR: ARIHANT GAUR
# ORGANIZATION: IvLabs, VNIT


import cv2
import numpy as np

def ImageRectification(pts1, pts2, K, image1, image2):
	E, mask = cv2.findEssentialMat(pts1, pts2, K, method = cv2.RANSAC, prob = 0.999, threshold = 0.4, mask = None)
	pts1 = pts1[mask.ravel() ==1]
	pts2 = pts2[mask.ravel() ==1]
	_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
	r1 = t / np.sqrt(np.matmul(t.T, t)[0][0])
	r2 = np.array([[-t[1], t[0], 0]])/np.sqrt(t[0]**2 + t[1]**2)
	print(r1, r2)
	r3 = np.cross(r1,r2.T)

	

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

# PERFORM IMAGE ENHANCEMENT TECHNIQUES

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
	if m.distance < 0.70 * n.distance:
		good.append(m)
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)
		
pts1 = np.int32(pts1)
pts2 = np.int32(pts2)


# Fundamental Matrix Calculation
#F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC)
#pts1 = pts1[mask.ravel() == 1]
#pts2 = pts2[mask.ravel() == 1]



imgLrec, imgRrec = ImageRectification(pts1, pts2, KL, imgL, imgR)
#print(len(pts1), len(pts2))

#imgL, imgR = DisplayEpipolarLines(imgL, imgR, pts1, pts2)

cv2.imshow('Left Image',imgL)
cv2.imshow('Right Image', imgR)
cv2.waitKey(0)
cv2.destroyAllWindows()
