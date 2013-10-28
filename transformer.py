import cv2
import numpy as np
import time

from main_utils import LATTICE_X, LATTICE_Y

absd = cv2.absdiff

HSV_X1 = 80
HSV_X2 = 600
HSV_Y1 = 80
HSV_Y2 = 400

def show_hsv(img, x1, x2, y1, y2):
    """ draws lines of roi in the image """
    cv2.line(img, (x1, y1),(x1, y2),(255,0,0))
    cv2.line(img, (x1, y1),(x2, y1),(255,0,0))
    cv2.line(img, (x2, y2),(x2, y1),(255,0,0))
    cv2.line(img, (x2, y2),(x1, y2),(255,0,0))

class Transformer:

    def __init__(self, height=480, width=640):
        self.last = np.zeros((height, width), np.uint8)
        self.element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        self.width = width
        self.height = height
        
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
        
        
    def skin_color_cue(self, img, box):
        element = self.element
        x1, x2, y1, y2 = box
        #x1_use = x1 < HSV_X1 and x1 or HSV_X1
        #x2_use = x2 > HSV_X2 and x2 or HSV_X2
        #y1_use = y1 < HSV_Y1 and y1 or HSV_Y1
        #y2_use = y2 > HSV_Y2 and y2 or HSV_Y2
        x1_use = 0; x2_use = 640; y1_use = 0; y2_use = 480
        result = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        img_hsv = cv2.cvtColor(img[y1_use:y2_use, x1_use:x2_use], cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_hsv)
        #daylight 6 - 20
        #artificial light 145 - 200 or 6 - 20 / 2-20 ??
        d = cv2.inRange(h, np.array([145],np.uint8), 
                           np.array([200],np.uint8))
        d = cv2.erode(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        result[y1_use:y2_use, x1_use:x2_use] = d
        #show_hsv(result, x1_use, x2_use, y1_use, y2_use)
        cv2.imshow('SKIN CUE', result)
        return result
        
    def edge_cue(self, img):
        """ATTENTION: too slow"""
        blurKsize = 5
        edgeKsize = 5
        img2 = cv2.medianBlur(img, blurKsize)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        cv2.Laplacian(gray, cv2.cv.CV_8U, gray, ksize = edgeKsize)
        return gray
        
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
        
        
        