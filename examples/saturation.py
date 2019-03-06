import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

# Read in an image, you can also try test1.jpg or test4.jpg
img = mpimg.imread('../test_images/test1.jpg') 

# Define a function that takes an image, gradient orientation,
# and threshold min / max values.
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, sobel_thresh=(0, 255)):
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
    binary_output[(scaled_sobel >= sobel_thresh[0]) & (scaled_sobel <= sobel_thresh[1])] = 1

    # Return the result
    return binary_output

# Define a function to return the magnitude of the gradient
# for a given sobel kernel size and threshold values
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
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
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
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
    # 2) Apply a threshold to the S channel
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1]) & (L > 50)] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def combine_threshs(grad_x, grad_y, mag_binary, dir_binary, col_binary, ksize=15):
    # Combine the previous thresholds
    combined = np.zeros_like(dir_binary)
    combined[((grad_x == 1) & (grad_y == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | (col_binary == 1)] = 1

    return combined    

    
# Plot the result
grad_x = abs_sobel_thresh(img, orient='x', sobel_kernel=15, thresh=(30, 100))
#grad_x_binary_edge = np.dstack((grad_x, grad_x, grad_x))*255
#plt.imshow(grad_x_binary_edge)


# Plot the result
grad_y = abs_sobel_thresh(img, orient='y', sobel_kernel=15, thresh=(30, 100))
#grad_y_edge = np.dstack((grad_y, grad_y, grad_y))*255
#plt.imshow(grad_y_edge)

# Plot the result
mag_binary = mag_thresh(img, sobel_kernel=15, thresh=(70, 100))
#mag_binary_edge = np.dstack((mag_binary, mag_binary, mag_binary))*255
#plt.imshow(mag_binary_edge)
    
# Plot the result
dir_binary = dir_threshold(img, sobel_kernel=15, thresh=(0.7, 1.3))
#dir_binary_edge = np.dstack((dir_binary, dir_binary, dir_binary))*255
#plt.imshow(dir_binary_edge)  
    
# Plot the result
hls_binary = hls_select(img, thresh=(170, 255))
#hls_binary_edge = np.dstack((hls_binary, hls_binary, hls_binary)) *255 # Grayscale to 3 dimension
#plt.imshow(hls_binary_edge)
#mpimg.imsave('../output_images/test1.jpg', hls_binary_edge)
    
# Run the function
combined = combine_threshs(grad_x, grad_y, mag_binary, dir_binary, hls_binary, ksize=15)

# Plot the result
combined_edge = np.dstack((combined, combined, combined)) *255 # Grayscale to 3 dimension
plt.imshow(combined_edge)