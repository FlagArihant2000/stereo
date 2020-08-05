import cv2
import numpy as np


cv2.namedWindow('window',cv2.WINDOW_NORMAL)
cv2.namedWindow('disparity', cv2.WINDOW_NORMAL)

disparity = cv2.pyrDown(cv2.imread('disp0.jpg',0))

#cv2.destroyAllWindows()
image = cv2.pyrDown(cv2.imread('im0.png'))
#print(image.shape)
b = 193.001/2
f = 3979.911/2
Q = np.array([[1, 0, 0, -2964/2], [0, 1, 0, -2000/2],[0, 0, 0, f],[0, 0, -1/b, -124.343/b]])

points = cv2.reprojectImageTo3D(disparity, Q)
mask = disparity > disparity.min()
#print(mask[0])
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
with open('stereoGT.ply', 'w') as f:
	f.write(ply_header %dict(vert_num = len(verts)))
	np.savetxt(f, verts, '%f %f %f %d %d %d')

print("DONE")
cv2.imshow('disparity', disparity)
cv2.imshow('window', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
