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

## Result
![stereo](https://user-images.githubusercontent.com/45517467/87001892-2a982e80-c1d6-11ea-9f7d-9b4716ce4f38.png)

Currently the results are being improved.
