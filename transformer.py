import cv2
import numpy as np
import time

from main_utils import LATTICE_X, LATTICE_Y

absd = cv2.absdiff

class Transformer:

    def __init__(self, light="Day", height=480, width=640):
        self.last = np.zeros((height, width), np.uint8)
        self.element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        self.light = light
        self.width = width
        self.height = height
        self.hsv = [1, 2, 145, 200]

    def set_color_ranges(self, color_ranges):
        self.hsv = color_ranges
        
    def smart_and(self, fst, sec):
        result = np.zeros((self.height, self.width), np.uint8)
        w = LATTICE_X
        h = LATTICE_Y
        rw = self.width/w
        rh = self.height/h
        for i in range(0,rw):
            for j in range(0,rh):
                roi_fst = fst[j*h:(j+1)*h, i*w:(i+1)*w]
                roi_sec = sec[j*h:(j+1)*h, i*w:(i+1)*w]
                final = cv2.bitwise_and(roi_fst, roi_sec)
                numb = cv2.countNonZero(final)
                if numb > 25:
                    #result[j*h:(j+1)*h, i*w:(i+1)*w] = cv2.bitwise_or(roi_fst, roi_sec)
                    result[j*h:(j+1)*h, i*w:(i+1)*w] = 255
        return result
        
    def _morpho_day(self, img):
        element = self.element
        # Day morpho: erode, 4xdilate or 2xerode, 4xdilate
        d = cv2.erode(img, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        return d

    def _morpho_daydim(self, img):
        element = self.element
        d = cv2.dilate(img, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        return d

    def _morpho_night(self, img):
        element = self.element
        # 2x erode
        d = cv2.erode(img, element)
        d = cv2.erode(d, element)
        # dilate
        d = cv2.dilate(d, element)
        # erode
        d = cv2.erode(d, element)
        # dilate
        d = cv2.dilate(d, element)
        # 3x erode
        d = cv2.erode(d, element)
        d = cv2.erode(d, element)
        d = cv2.erode(d, element)
        # 3x dilate
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        return d

    def skin_color_cue(self, img):
        result = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_hsv)
        d = cv2.inRange(h, np.array(self.hsv[2], np.uint8), 
                           np.array(self.hsv[3], np.uint8))
        d2 = cv2.inRange(h, np.array(self.hsv[0], np.uint8), 
                            np.array(self.hsv[1], np.uint8))
        d = cv2.bitwise_or(d, d2)
        if self.light == "Night":
            d = self._morpho_night(d)
        elif self.light == "Day":
            d = self._morpho_day(d)
        elif self.light == "DayDim":
            d = self._morpho_daydim(d)
        result = d
        
        cv2.imshow('SKIN CUE', result)
        return result
        
    def move_cue(self, img):
        element = self.element
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result2 = absd(gray, self.last)
        dummy, result3 = cv2.threshold(result2, 9, 255, cv2.THRESH_BINARY)
        eroded = cv2.erode(result3, element)
        dilated = cv2.dilate(eroded, element)
        dilated = cv2.dilate(dilated, element)
        dilated = cv2.dilate(dilated, element)
        self.last = gray
        return dilated
        
    def postprocess(self, img):
        img = cv2.medianBlur(img, 3)
        return img
        
        
        