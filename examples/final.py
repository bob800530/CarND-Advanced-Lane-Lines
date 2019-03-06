import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import glob

# Use chessboard data to create transform matrix
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

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, sobel_kernel))
    
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the magnitude of the gradient
# and applies a threshold
def mag_thres(img, sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale    
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function that applies Sobel x and y, 
# then computes the direction of the gradient
# and applies a threshold.
def dir_thresh(img, sobel_kernel=9, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return the binary image
    return binary_output

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    L = hls[:,:,1]
    S = hls[:,:,2]
    # 2) Apply a threshold to the S channel and L-Channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1]) & (L > 50)] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def combine_threshs(grad_x, grad_y, mag_binary, dir_binary, col_binary, ksize=15):
    # Combine the previous thresholds
    combined = np.zeros_like(dir_binary)
    combined[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (col_binary == 1)] = 1

    return combined    


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(binary_warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        ### TO-DO: Find the four below boundaries of the window ###
        win_xleft_low = leftx_current - margin  # Update this
        win_xleft_high = leftx_current + margin  # Update this
        win_xright_low = rightx_current - margin  # Update this
        win_xright_high = rightx_current + margin  # Update this
        
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),
        (win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),
        (win_xright_high,win_y_high),(0,255,0), 2) 
        
        ### TO-DO: Identify the nonzero pixels in x and y within the window ###
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

#    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        print("find point error")
        pass

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty, out_img


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    leftx, lefty, rightx, righty, out_img = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit` ###
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    try:
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fit = [1,1,0]
        right_fit = [1,1,0]
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty
            
    ## Visualization ##
    # Colors in the left and right lane regions
    out_img[lefty, leftx] = [255, 0, 0]
    out_img[righty, rightx] = [0, 0, 255]
       
    # Draw the lane and region onto the warped blank image
    line_img = np.zeros_like(out_img)
    window_img = np.zeros_like(out_img)
    line_img[lefty, leftx] = [255, 0, 0]
    line_img[righty, rightx] = [0, 0, 255]

    left_line_window = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_line_window = np.array([np.flipud(np.transpose(np.vstack([right_fitx, 
                                  ploty])))])
    left_line_pts = np.hstack((left_line_window, right_line_window))
    
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0)) 

    # Draw left and right lines
    for index in range(binary_warped.shape[0]):
        cv2.circle(out_img, (int(left_fitx[index]), int(ploty[index])), 3, (255,255,0))
        cv2.circle(out_img, (int(right_fitx[index]), int(ploty[index])), 3, (255,255,0))                   
    region_img = cv2.addWeighted(line_img, 1, window_img, 0.3, 0)  
    # Plot the results
    # Visualize undirstorsion
#    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
#    ax1.imshow(line_img)
#    ax2.imshow(out_img)

    return region_img,left_fit,right_fit,left_fitx,right_fitx,leftx, lefty, rightx, righty,ploty
    

corners = np.zeros((4,1,2), np.float32) 
corners[0] = [550,480]
corners[1] = [730,480]
corners[2] = [220,700]
corners[3] = [1080,700]
src = np.float32(
            [[280,  700],  # Bottom left
             [595,  460],  # Top left
             [725,  460],  # Top right
             [1125, 700]]) # Bottom right
        
dst = np.float32(
            [[250,  720],  # Bottom left
             [250,    0],  # Top left
             [1065,   0],  # Top right
             [1065, 720]]) # Bottom right   


offset =50
def perspective(img):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size) 
    
    return warped
    
def unperspective(img):
    img_size = (img.shape[1], img.shape[0])
    M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size) 
    
    return warped    
    
def mark_lanes(img, region_real_img):
    left_lane = region_real_img[:,:,0]
    right_lane = region_real_img[:,:,2]
    img[left_lane>0] = [255,0,0]
    img[right_lane>0] = [0,0,255]
    return img

def add_curvature(img, left_fit, right_fit, left_fitx, right_fitx,leftx, lefty, rightx, righty):
    
    
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y
    ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
    # Fit a second order polynomial to pixel positions in each fake lane line
    left_fit = np.polyfit(ploty, left_fitx, 2)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fit = np.polyfit(ploty, right_fitx, 2)
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 25/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/800 # meters per pixel in x dimension

    # Fit new polynomials to x,y in world space
    y_eval = np.max(ploty)
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    
    left_curverad = ("%.2f" % left_curverad)
    curvature_word = 'Radius of Curvature = ' + str(left_curverad) + '(m)'
    
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    road_mid = img.shape[1]/2    
    car_mid = (right_fitx[719] + left_fitx[719])/2
    
    xm_per_pix = 3.7/(right_fitx[719] - left_fitx[719]) # meters per pixel in x dimension
    mid_dev = car_mid - road_mid;
    mid_dev_meter = abs(mid_dev)*xm_per_pix
    mid_dev_meter = ("%.2f" % mid_dev_meter)
    if road_mid>0:
        mid_dev_word = 'Vehicle is '+ str( mid_dev_meter)+ 'm right of center'
    else:
        mid_dev_word = 'Vehicle is '+ str(mid_dev_meter)+ 'm left of center'
    word_img = cv2.putText(img,curvature_word,(100,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
    word_img = cv2.putText(img,mid_dev_word,(100,160),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
    return word_img


ret, mtx, dist, rvecs, tvecs = calibrate_Camera()  
 
# Read image
img = cv2.imread('../test_images/test6.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img = cv2.undistort(img, mtx, dist, None, mtx)

grad_x = abs_sobel_thresh(img, orient='x', sobel_kernel=15, thresh=(30, 100))

grad_y = abs_sobel_thresh(img, orient='y', sobel_kernel=15, thresh=(30, 100))

mag_binary = mag_thres(img, sobel_kernel=15, thresh=(50, 100))

dir_binary = dir_thresh(img, sobel_kernel=15, thresh=(0.7, 1.3))

hls_binary = hls_select(img, thresh=(170, 255))
    
# Run the function
combined = combine_threshs(grad_x, grad_y, mag_binary, dir_binary, hls_binary, ksize=15)

warped =  perspective(combined)  

region_img,left_fit,right_fit,left_fitx,right_fitx,leftx, lefty, rightx, righty,ploty = fit_polynomial(warped)
region_real_img =  unperspective(region_img)
marked_img = mark_lanes(img, region_real_img)
region_real_img = region_real_img.astype(np.uint8)
weighted_img = cv2.addWeighted(marked_img, 1, region_real_img, 0.5, 0)   
word_img = add_curvature(weighted_img, left_fit, right_fit, left_fitx, right_fitx,leftx, lefty, rightx, righty)

plt.imshow(word_img)
#plt.imshow(warped)
mpimg.imsave('../output_images/processed_images/processed_test6.jpg', word_img)