
import numpy as np
import cv2
import matplotlib.pyplot as plt


# Read image
img = cv2.imread('../test_images/straight_lines1.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
img_size = (gray.shape[1], gray.shape[0])
# Find the corners of lanes
# If found, add object points, image points
corners = np.zeros((4,1,2), np.float32) 
corners[0] = [550,480]
corners[1] = [730,480]
corners[2] = [220,700]
corners[3] = [1080,700]
src = np.float32([corners[0], corners[1], corners[3], corners[2]])

offset = 50
# For destination points, I'm arbitrarily choosing some points to be
# a nice fit for displaying our warped result 
# again, not exact, but close enough for our purposes
dst = np.float32([[offset*4, 0], [img_size[0]-offset*4, 0], 
                             [img_size[0]-offset*4, img_size[1]], 
                             [offset*4, img_size[1]]])
                             
# Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, img_size)

# Add polygon to mark two lane lines
pts = np.array([corners[0], corners[1], corners[3], corners[2]], np.int32)
cv2.polylines(img,[pts],True,(0,255,255),3)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=50)

# Add a perspective polygon to mark two lane lines
pts_warped = np.array([dst[0], dst[1], dst[2], dst[3]], np.int32)
cv2.polylines(warped,[pts_warped],True,(0,255,255),3)

ax2.imshow(warped)
ax2.set_title('Perspective Image', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=1, bottom=0.)