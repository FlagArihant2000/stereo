# Stereo

The objective is to reconstruct a 3D equivalent, given the pair of images.

## Run

1. ```git clone https://github.com/FlagArihant2000/stereo.git```
2. ```cd stereo```
3. Execute ```python3 stereo.py```
4. Result is obtained in ```stereo.ply```, which can be viewed using MeshLab, Open3D, or any other related software or library to view point clouds.

## Working

1. Acquire left and right image.
2. Feature Extraction using SIFT.
3. Feature Matching using Brute Force KNN and rejecting outliers.
4. Finding essential matrix of point correspondance and reject outliers using RANSAC.
5. Recover Pose.
6. Perform stereo rectification using the recovered poses. The intrinsic camera matrix and distortion coefficients are known and present in ```calib.txt```.
7. Calculate the Disparity map with Semi Global Block Matching (SGBM). Number of disparities are mentioned in ```calib.txt```, whereas the other parmeter are taken in the code itself.
8. Finding left and right disparity map, followed by performing Weighted Least Square (WLS Filtering). The filtered image is considered as the final disparity map.
9. Defining reprojection matrix Q to get the 3D equivalent of disparity matrix.
10. Generating the point cloud, which is stored in ```stereo.ply```.

## Code Parameters

1. Disparity Map with SGBM
```
max_disparity = 199
min_disparity = 23
num_disparities = max_disparity - min_disparity
window_size = 5
stereo = cv2.StereoSGBM_create(minDisparity = min_disparity, numDisparities = num_disparities, blockSize = 5, uniquenessRatio = 5, speckleWindowSize = 5, speckleRange = 5, disp12MaxDiff = 2, P1 = 8*3*window_size**2, P2 = 32*3*window_size**2)
```
2. WLS Filter
```
lamb = 8000
sig = 1.5
visual_multiplier = 1.0
wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
wls_filter.setLambda(lamb)
wls_filter.setSigmaColor(sig)
```

## Result
![stereo](https://user-images.githubusercontent.com/45517467/87001892-2a982e80-c1d6-11ea-9f7d-9b4716ce4f38.png)

My Point Cloud: [Click](stereo.ply)
Ground Truth Point Cloud: [Click](https://drive.google.com/file/d/1gB1SkUjDp1Rdh9CE5o9vZe9EMabqKp5-/view?usp=sharing)

Currently, this concept is being extended to Multiview stereo reconstruction.
