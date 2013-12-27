import cv2
import numpy as np
import time

from main_utils import LATTICE_X, LATTICE_Y
from main_utils import draw_rects
absd = cv2.absdiff

def rev_area(rect):
    return 1/float(rect[2]*rect[3])

class Transformer:

    def __init__(self, light, color_h, color_yv, threshold, height=480, width=640):
        self.last = np.zeros((height, width), np.uint8)
        self.element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        self.light = light
        self.width = width
        self.height = height
        self.hsv = color_h
        self.yuv = color_yv
        self.thr = threshold
        
    def _morpho_day(self, img):
        element = self.element
        d = cv2.erode(img, element)
        d = cv2.erode(img, element)
        d = cv2.erode(img, element)
        d = cv2.erode(img, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        return d

    def _morpho_night(self, img):
        element = self.element
        d = cv2.erode(img, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        return d

    def skin_color_cue(self, img):
        result = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        h, s, v = cv2.split(img_hsv)
        dummy, value = cv2.threshold(v, self.thr, 255, cv2.THRESH_BINARY)
        y, u, yv = cv2.split(img_yuv)
        r1_h = cv2.inRange(h, np.array(self.hsv[2], np.uint8), 
                           np.array(self.hsv[3], np.uint8))
        r2_h = cv2.inRange(h, np.array(self.hsv[0], np.uint8), 
                           np.array(self.hsv[1], np.uint8))
        r1_yv = cv2.inRange(yv, np.array(self.yuv[2], np.uint8), 
                            np.array(self.yuv[3], np.uint8))
        r2_yv = cv2.inRange(yv, np.array(self.yuv[0], np.uint8), 
                            np.array(self.yuv[1], np.uint8))

        whole_h = cv2.bitwise_or(r1_h, r2_h)
        #cv2.imshow('h', whole_h)
        whole_yv = cv2.bitwise_or(r1_yv, r2_yv)
        #cv2.imshow('yv', whole_yv)
        d = cv2.bitwise_and(whole_yv, whole_h)
        #cv2.imshow('value', value)
        d = cv2.bitwise_and(d, value)   
        #cv2.imshow('whole', d)
        if self.light == "Night":
            d = self._morpho_night(d)
        elif self.light == "Day":
            d = self._morpho_day(d)
        result = d
        return result
        
    def move_cue(self, img):
        element = self.element
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result2 = absd(gray, self.last)
        dummy, result3 = cv2.threshold(result2, 9, 255, cv2.THRESH_BINARY)
        eroded = cv2.erode(result3, element)
        dilated = cv2.dilate(eroded, element)
        d = dilated.copy()
        contours, hier = cv2.findContours(d, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = self.choose_contour(contours)
        big_rect = [640,480,0,0]
        for rect in rects:
            if rect[0] < big_rect[0]:
                big_rect[0] = rect[0]
            if rect[1] < big_rect[1]:
                big_rect[1] = rect[1]
            if rect[0] + rect[2] > big_rect[0] + big_rect[2]:
                big_rect[2] = rect[0] + rect[2] - big_rect[0]
            if rect[1] + rect[3] > big_rect[1] + big_rect[3]:
                big_rect[3] = rect[1] + rect[3] - big_rect[1]
        if big_rect[2]*big_rect[3] < 16000:
            big_rect = [0, 10, 640, 470]
        elif big_rect[2]*big_rect[3] < 24000:
            big_rect[0] = max(0, big_rect[0] - 20)
            big_rect[2] += 50
            big_rect[1] = max(0, big_rect[1] - 25)
            big_rect[3] += 90
        elif big_rect[2]*big_rect[3] < 28000:
            big_rect[0] = max(0, big_rect[0] - 15)
            big_rect[2] += 30
            big_rect[1] = max(0, big_rect[1] - 20)
            big_rect[3] += 80
        elif big_rect[2] > 1.5*big_rect[3]:
            big_rect[3] += 50
        elif big_rect[3] > 1.5*big_rect[2]:
            big_rect[2] += 50
        
        result = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        result[big_rect[1]:big_rect[1]+big_rect[3], big_rect[0]:big_rect[0]+big_rect[2]] = 255
        #cv2.imshow('MOVE', result)
        self.last = gray
        return result

    def choose_contour(self, contours):
        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 200]
        s_rects = sorted(rects, key=rev_area)
        return rects[:20]
        
    def postprocess(self, img):
        img = cv2.medianBlur(img, 3)
        return img
        
        
        