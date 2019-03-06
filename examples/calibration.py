import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def calibrate_Camera():
    images_path = glob.glob('../camera_cal/calibration*.jpg')
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    for img_path in images_path:
        img = cv2.imread(img_path)
        # Read calibration image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6),None)        
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    # Use cv2.calibrateCamera()
    # Do camera calibration given object points and image points
    img_size = (img.shape[1], img.shape[0])
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    return ret, mtx, dist, rvecs, tvecs

ret, mtx, dist, rvecs, tvecs = calibrate_Camera()

# Use CalibrateCamera() mtx to undistort calibration3.jpg
img = cv2.imread('../camera_cal/calibration3.jpg')
undistorted = cv2.undistort(img, mtx, dist, None, mtx)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)
ax2.imshow(undistorted)
ax2.set_title('Undistorted Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=1, bottom=0.)