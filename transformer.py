import cv2
import numpy as np
import time

from main_utils import LATTICE_X, LATTICE_Y
from main_utils import draw_rects
from cleaner import Cleaner
from config import CLASSIFIER, HEIGHT, WIDTH


absd = cv2.absdiff

def rev_area(rect):
    return 1/float(rect[2]*rect[3])

class Transformer:

    def __init__(self, light, color_h, color_yv, threshold):
        self.last = np.zeros((HEIGHT, WIDTH), np.uint8)
        self.element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        self.hsv = color_h
        self.yuv = color_yv
        self.thr = threshold
        self.cleaner = Cleaner(light)
        self.skin_classifier = getattr(self, "%s_skin_classifier" % CLASSIFIER)

    def turn_on_bayes_classifier(self, s_h, s_v):
        self.s_h = s_h
        self.s_v = s_v

    def find_important_planes(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        h_, s_, v_ = cv2.split(hsv)
        y, u, v = cv2.split(yuv)
        return h_, v_, v

    def linear_skin_classifier(self, img):
        h_, v_, v = self.find_important_planes(img)

        dummy, value = cv2.threshold(v_, self.thr, 255, cv2.THRESH_BINARY)
        r1_h = cv2.inRange(h_, np.array(self.hsv[2], np.uint8),
                           np.array(self.hsv[3], np.uint8))
        r2_h = cv2.inRange(h_, np.array(self.hsv[0], np.uint8),
                           np.array(self.hsv[1], np.uint8))
        r1_v = cv2.inRange(v, np.array(self.yuv[2], np.uint8),
                           np.array(self.yuv[3], np.uint8))
        r2_v = cv2.inRange(v, np.array(self.yuv[0], np.uint8),
                           np.array(self.yuv[1], np.uint8))

        whole_h = cv2.bitwise_or(r1_h, r2_h)
        whole_v = cv2.bitwise_or(r1_v, r2_v)
        d = cv2.bitwise_and(whole_v, whole_h)
        d = cv2.bitwise_and(d, value)
        return d

    def classify(self, img, plane):
        if plane == "v":
            img = self.s_v[img]
        if plane == "h":
            img = self.s_h[img]
        return img

    def bayes_skin_classifier(self, img):
        h_, v_, v = self.find_important_planes(img)

        h_class = self.classify(h_, "h")
        v_class = self.classify(v, "v")
        d = cv2.bitwise_and(h_class, v_class)
        return d
        
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
        big_rect = [0, 10, 640, 470]
        
        result = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        result[big_rect[1]:big_rect[1]+big_rect[3], big_rect[0]:big_rect[0]+big_rect[2]] = 255
        #cv2.imshow('MOVE', result)
        self.last = gray
        return result

    def choose_contour(self, contours):
        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 200]
        s_rects = sorted(rects, key=rev_area)
        return s_rects[:20]
        
    def clean_whole_image(self, img):
        return self.cleaner.clean(img)
        
        