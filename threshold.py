import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Thresholding:
            
    # Define a function that takes an image, gradient orientation,
    # and threshold min / max values.
    def abs_sobel_thresh(self, img, orient = 'x', thresh_min = 0, thresh_max = 255):
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Apply x or y gradient with the OpenCV Sobel() function
        # and take the absolute value
        if orient == 'x':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
        elif orient == 'y':
            abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
        # Rescale back to 8 bit integer
        scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
        # Create a copy of the image
        binary = np.zeros_like(scaled_sobel)
        # Apply the threshold
        binary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
        return binary

    def thresholding(self, img, orient, s_thresh=(170, 255), sx_thresh=(30, 255)):   
        # Sobel x 
        sobel_x = abs_sobel_thresh(self, img, orient, 30, 255)

        # White and yellow from RGB
        rgb_white = cv2.inRange(img, (200, 200, 200), (255, 255, 255))
        rgb_yellow = cv2.inRange(img, (20, 100, 100), (50, 255, 255))

        # White and yellow from HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_white = cv2.inRange(hsv, (200, 200, 200), (255, 255, 255))
        hsv_yellow = cv2.inRange(hsv, (20, 100, 100), (50, 255, 255))

        # S channel from HLS
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        s = hls[:,:,2]
        s_output = np.zeros_like(s)
        s_output[(s > 170) & (s <= 255)] = 1

        # White and yellow from HLS
        hls_white = cv2.inRange(hls, (200,200,200), (255,255,255))
        hls_yellow = cv2.inRange(hls, (20, 100, 100), (50, 255, 255))

        masked = sobel_x | s_output | rgb_white | rgb_yellow | hsv_white | hsv_yellow | hls_white | hls_yellow
        return masked