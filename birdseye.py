import os
import numpy as np
import cv2

class BirdsEye:
    
    def __init__(self, src, dst):
        self.src = src
        self.dst = dst
    
    # Function to get the M and M inverse matrix for perspective transform
    def PerspectiveTransform(self):
        M = cv2.getPerspectiveTransform(self.src, self.dst)
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)
        return M, Minv