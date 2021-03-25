import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class CameraCalibration:

    # Function to get object points given number of corners 
    # in x and y directions
    def get_objp(self, nx, ny):
        objp = np.zeros((nx*ny,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
        return objp

    # Function to calibrate camera given calibration images
    def camera_calibration(self, images, nx, ny, draw = False):
        objpoints = []
        imgpoints = [] 
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Try nx and ny first, then reduces the numbers if not found
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            objp = self.get_objp(nx, ny)
            if ret is False:
                ret, corners = cv2.findChessboardCorners(gray, (nx - 1, ny), None)
                objp = self.get_objp(nx - 1, ny)
            if ret is False:
                ret, corners = cv2.findChessboardCorners(gray, (nx, ny - 1), None)
                objp = self.get_objp(nx, ny - 1)
            if ret is False:
                ret, corners = cv2.findChessboardCorners(gray, (nx - 1, ny - 1), None)
                objp = self.get_objp(nx - 1, ny - 1)
            if ret is False:
                ret, corners = cv2.findChessboardCorners(gray, (nx - 2, ny), None)
                objp = self.get_objp(nx - 2, ny)
            if ret is False:
                ret, corners = cv2.findChessboardCorners(gray, (nx, ny - 2), None)
                objp = self.get_objp(nx, ny - 2)
            if ret is False:
                ret, corners = cv2.findChessboardCorners(gray, (nx - 2, ny - 2), None)
                objp = self.get_objp(nx - 2, ny - 2)

            if ret is True:
                imgpoints.append(corners)
                objpoints.append(objp)
                # If we want to draw the corners
                if draw is True:
                    img = cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
                    plt.figure(figsize=(5,4))
                    plt.imshow(img)
                    plt.show()
            else:
                print('Could not find corners from %s' %fname)
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        return mtx, dist