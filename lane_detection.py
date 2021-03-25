import os
import numpy as np
import cv2

class LaneDetection :
    
    def __init__(self, ym_per_pix, xm_per_pix, Minv):
        # Define conversions in x and y from pixels space to meters
        self.ym_per_pix = ym_per_pix # meters per pixel in y dimension
        self.xm_per_pix = xm_per_pix # meteres per pixel in x dimension
        self.Minv = Minv

    # Function to find max value and index for an array from left to right
    def FindMaxInd(self,array, left, right):
        max_value, max_ind = 0, 0
        for i in range(left, right):
            if array[i] > max_value:
                max_value, max_ind = array[i], i
        return max_value, max_ind

    # Function to reject the point outliers by comparing to the median value
    def RejectOutlier(self,x_array, y_array):
        median = np.median(x_array)
        dev = x_array - median
        ind = []
        for i, x in enumerate(dev):
            if abs(x) > 200:
                ind.append(i)
        x_array = np.delete(x_array, ind)
        y_array = np.delete(y_array, ind)
        return x_array, y_array

    # Function to find the lines from the points
    def FindLines(self,img, undist, step):
        left = []
        right = []
        center = int(img.shape[1]/2)
        # Create an image to draw the lines on
        warp_zero = np.zeros_like(img).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        for i in range(img.shape[1] - step, 0, -step):
            histogram = np.sum(img[i:i+step,:], axis = 0)
            left_max_var, left_max_ind = self.FindMaxInd(histogram, 0, center)
            right_max_var, right_max_ind = self.FindMaxInd(histogram, center, img.shape[1])
            if(left_max_var > 0):
                left.append((left_max_ind, int((i+i+step)/2)))
            if(right_max_var > 0):
                right.append((right_max_ind, int((i+i+step)/2)))

        left_x = np.array([item[0] for item in left])
        left_y = np.array([item[1] for item in left])
        left_x, left_y = self.RejectOutlier(left_x, left_y)
        left_fit = np.polyfit(left_y, left_x, 2)
        left_y_ext = np.append(left_y, 0)
        left_y_ext = np.append(720, left_y_ext)
        left_fitx = left_fit[0]*left_y_ext**2 + left_fit[1]*left_y_ext + left_fit[2]

        right_x = np.array([item[0] for item in right])
        right_y = np.array([item[1] for item in right])
        right_x, right_y = self.RejectOutlier(right_x, right_y)
        right_fit = np.polyfit(right_y, right_x, 2)
        right_y_ext = np.append(right_y, 0)
        right_y_ext = np.append(720, right_y_ext)
        right_fitx = right_fit[0]*right_y_ext**2 + right_fit[1]*right_y_ext + right_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, left_y_ext]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_y_ext])))])
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
         # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.Minv, (img.shape[1], img.shape[0])) 
        # Combine the result with the original image
        result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
        yval = img.shape[1]
        left_fit_cr = np.polyfit(left_y*self.ym_per_pix, left_x*self.xm_per_pix, 2)
        right_fit_cr = np.polyfit(right_y*self.ym_per_pix, right_x*self.xm_per_pix, 2)
        left_curverad = ((1 + (2*left_fit_cr[0]*yval + left_fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*yval + right_fit_cr[1])**2)**1.5) \
                                        /np.absolute(2*right_fit_cr[0])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'Left line curve: %dm' % left_curverad, (50,50), font, 1,(255,255,255),3)
        cv2.putText(result, 'Right line curve: %dm' % right_curverad, (50,100), font, 1,(255,255,255),3)

        lane_middle = left_fitx[0] + (right_fitx[0] - left_fitx[0])/2.0
        deviation = (lane_middle - 640)*self.xm_per_pix
        if deviation >= 0:
            cv2.putText(result, 'Vehicle is %.2fm right of center' % deviation, (50,150), font, 1,(255,255,255),3)
        else:
            cv2.putText(result, 'Vehicle is %.2fm left of center' % -deviation, (50,150), font, 1,(255,255,255),3)
        #ax[0].scatter(left_x, left_y, c = 'r')
        #ax[0].scatter(right_x, right_y, c = 'b')
        #ax[0].plot(left_fitx, left_y_ext, color='green', linewidth=3)
        #ax[0].plot(right_fitx, right_y_ext, color='green', linewidth=3)

        return result, left_x, left_y, right_x, right_y, left_fitx, left_y_ext, right_fitx, right_y_ext