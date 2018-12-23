import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 50
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

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
    except ValueError:
        # Avoids an error if the above is not implemented fully
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
    region_img = cv2.addWeighted(line_img, 1, window_img, 0.2, 0)  

    return region_img,left_fit,right_fit,left_fitx,right_fitx,ploty
    
def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:,:,0]
    L = hls[:,:,1]
    S = hls[:,:,2]
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
 
def perspective(img):
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    # Find the corners of lanes
    # If found, add object points, image points
    corners = np.zeros((4,1,2), np.float32) 
    corners[0] = [530,450]
    corners[1] = [750,450]
    corners[2] = [80,700]
    corners[3] = [1200,700]
    src = np.float32([corners[0], corners[1], corners[3], corners[2]])
    
    offset = 0
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([[offset*4, 0], [img_size[0]-offset*4, 0], 
                                 [img_size[0]-offset*4, img_size[1]], 
                                 [offset*4, img_size[1]]])
                             
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, img_size) 
    
    return warped
    
def unperspective(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_size = (gray.shape[1], gray.shape[0])
    # Find the corners of lanes
    # If found, add object points, image points
    corners = np.zeros((4,1,2), np.float32) 
    corners[0] = [530,450]
    corners[1] = [750,450]
    corners[2] = [80,700]
    corners[3] = [1200,700]
    src = np.float32([corners[0], corners[1], corners[3], corners[2]])
    
    offset = 0
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([[offset*4, 0], [img_size[0]-offset*4, 0], 
                                 [img_size[0]-offset*4, img_size[1]], 
                                 [offset*4, img_size[1]]])
                             
    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_size) 
    
    return warped    
    
def mark_lanes(img, region_real_img):
    left_lane = region_real_img[:,:,0]
    right_lane = region_real_img[:,:,2]
    img[left_lane>0] = [255,0,0]
    img[right_lane>0] = [0,0,255]
    return img

def add_curvature(img, left_fit, right_fit, left_fitx, right_fitx):
    y_eval = np.max(left_fit)
    ym_per_pix = 30/720 # meters per pixel in y dimensio
    # Calculation of R_curve (radius of curvature)
    left_curverad = ((1 + (2*left_fit[0]*y_eval*ym_per_pix + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*ym_per_pix + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])    
    #print(left_curverad, 'm', right_curverad, 'm')
    curvature_word = 'Radius of Curvature = ' + str(left_curverad) + '(m)'
    
    gray = cv2.cvtColor(weighted_img,cv2.COLOR_BGR2GRAY)
    road_mid = gray.shape[1]/2    
    car_mid = (right_fitx[719] + left_fitx[719])/2
    
    xm_per_pix = 3.7/(right_fitx[719] - left_fitx[719]) # meters per pixel in x dimension
    mid_dev = car_mid - road_mid;
    mid_dev_meter = abs(mid_dev)*xm_per_pix
    if road_mid>0:
        mid_dev_word = 'Vehicle is '+ str( mid_dev_meter)+ 'm right of center'
    else:
        mid_dev_word = 'Vehicle is '+ str( mid_dev_meter)+ 'm left of center'
    word_img = cv2.putText(weighted_img,curvature_word,(100,80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
    word_img = cv2.putText(weighted_img,mid_dev_word,(100,160),cv2.FONT_HERSHEY_SIMPLEX,1.5,(255,255,255),2)
    return word_img
    
# Read image
img = cv2.imread('../test_images/straight_lines2.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
warped =  perspective(img)  

hls_binary = hls_select(warped, thresh=(150, 255))*255

region_img,left_fit,right_fit,left_fitx,right_fitx,ploty = fit_polynomial(hls_binary)
region_real_img =  unperspective(region_img)
marked_img = mark_lanes(img, region_real_img)
weighted_img = cv2.addWeighted(marked_img, 1, region_real_img, 0.5, 0)   
word_img = add_curvature(weighted_img, left_fit, right_fit, left_fitx, right_fitx)
    
plt.imshow(word_img)
#plt.imshow(warped)
mpimg.imsave('../output_images/processed_straight_lines2.jpg', word_img)