"""
    TODO:
    -> wykrywanie gestu
    
    -> przepisanie do cythona
    -> testy sieciowe
    -> dokumentacja
    -> proby z innymi metodami + studia literaturowe

"""

import cv2
import os
import numpy as np
from collections import deque

from main_utils import draw_rects


def area(rect):
    return rect[2]*rect[3]

def get_roi(img, rect):
    x,y,w,h = rect
    roi = img[y:y+h, x:x+w]
    return roi

class EnviromentInfo(object):

    def __init__(self, middle_non_zero, rect_count, biggest_rect_area):
        self.middle_non_zero = middle_non_zero
        self.rect_count = rect_count
        self.biggest_rect_area = biggest_rect_area

    def points(self):
        if self.rect_count > 400:
            r = -30
        elif self.rect_count > 300:
            r = 0
        elif self.rect_count > 200:
            r = 8
        elif self.rect_count > 100:
            r = 20 + (200 - self.rect_count)/3
        elif 15 < self.rect_count < 100:
            r = 20 + (200 - self.rect_count)/2
        else:
            r = -30 
        if self.biggest_rect_area < 2000:
            p = -30
        elif self.biggest_rect_area < 20000:
            p = self.biggest_rect_area/2000
        elif self.biggest_rect_area < 60000:
            p = self.biggest_rect_area/1300
        elif self.biggest_rect_area < 120000:
            p = (self.biggest_rect_area - 60000)/2000
        else:
            p = 0
        if self.middle_non_zero < 2000:
            m = -30
        if self.middle_non_zero < 20000:
            m = self.middle_non_zero/2000
        elif self.middle_non_zero < 55000:
            m = self.middle_non_zero/1300
        elif self.middle_non_zero < 110000:
            m = (self.middle_non_zero - 55000)/4000
        else:
            m = 0
        #print m, r, p, "#"
        return m + r + p


class ThresholdInfo(object):

    def __init__(self, threshold, defects_count, prc_defect_area, avg_area, prc_avg_area):
        self.threshold = threshold
        self.defects_count = defects_count
        self.prc_defect_area = prc_defect_area
        self.avg_area = avg_area
        self.prc_avg_area = prc_avg_area

    def points(self):
        return 0


class CalibrationOld(object):
    """
        Calibrates HSV parameters and cut threashold for binary image
        to find best possible match with enviroment.
    """

    def __init__(self):
        self.img = None
        self.element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        
        self.choice = 6
        self.best_conf = [0, 0, 0, 0]
        self.middle_rect = [150, 100, 400, 350]
        self.conf_to_check = ([4, 20, 250, 250], [1, 2, 145, 200], [1, 2, 145, 190], [2, 20, 145, 190], [4, 20, 145, 200], [2, 20, 120, 190])
        self.conf_match = ["Day", "Night", "Night", "Day", "Day", "Day", "BadDecision"]
        self.enviroment_info = []
        self.threshold_info = []
        self.current_conf = self.conf_to_check[0]
        self.counter = 0
        
        
        self.phase = 0
        self.end = 0
        self.stabilize = 0
        self.threshold_start = 90
        self.threshold_end = 125
        self.threshold = self.threshold_start
        self.final_threshold = 40
        self.light = "Day"
        self.clear()

    def clear(self):
        self.current_rect_count = 0
        self.current_biggest_rect_area = 0
        self.current_middle_non_zero = 0

        self.defects_count = 0
        self.prc_defect_area = 0
        self.avg_area = 0

    def show_mode(self, img):
        element = self.element
        result = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_hsv)
        color_range = [2, 20, 145, 190]
        d = cv2.inRange(h, np.array(color_range[0],np.uint8), 
                           np.array(color_range[1],np.uint8))
        d2 = cv2.inRange(h, np.array(color_range[2],np.uint8), 
                            np.array(color_range[3],np.uint8))
        d = cv2.bitwise_or(d, d2)
        d = cv2.erode(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        result = d
        cv2.imshow('Result', result)

    def choose_contour(self, contours, threshold=200):
        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > threshold]
        def area(rect):
            return 1/float(rect[2]*rect[3])
        s_rects = sorted(rects, key=area)
        return s_rects[:2]

    def biggest_cnt(self, cnts):
        biggest = None
        biggest_area = 0
        for cnt in cnts:
            m = cv2.moments(cnt)
            if m["m00"] > biggest_area:
                biggest = cnt
                biggest_area = m["m00"]
        return biggest

    def smart_filter(self, hch, semi):
        h1 = cv2.inRange(hch, np.array([1],np.uint8), 
                             np.array([7],np.uint8))
        h2 = cv2.inRange(hch, np.array([145],np.uint8), 
                             np.array([200],np.uint8))
        hch = cv2.bitwise_or(h1, h2)
        hch = cv2.erode(hch, self.element)
        hch = cv2.erode(hch, self.element)
        hch = cv2.dilate(hch, self.element)
        h,w = hch.shape
        hp = 10; wp = 10
        for j in range(h/hp):
            for i in range(w/wp):
                part = hch[j*hp:(j+1)*hp, i*wp:(i+1)*wp]
                cnt = cv2.countNonZero(part)
                if cnt < 10:
                    semi[j*hp:(j+1)*hp, i*wp:(i+1)*wp] = 0
        return semi

    def update(self, img):
        if self.end:
            return
        if self.phase == 0:
            self.calibrate_hsv(img)
            #print "*",
        elif self.phase == 1:
            self.feedback()
            #print "*",
        elif self.phase == 2:
            self.feedback()
            #print "*",
        elif self.phase == 3:
            self.calibrate_threshold(img)
            print "*",
        elif self.phase == 4:
            self.feedback()
            #print "*"

    def calibrate_hsv(self, img):        
        element = self.element
        result = np.zeros((img.shape[0], img.shape[1]), np.uint8)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_hsv)
        d = cv2.inRange(h, np.array([self.current_conf[0]],np.uint8), 
                           np.array([self.current_conf[1]],np.uint8))
        d2 = cv2.inRange(h, np.array([self.current_conf[2]],np.uint8), 
                            np.array([self.current_conf[3]],np.uint8))
        d = cv2.bitwise_or(d, d2)
        d = cv2.erode(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        d = cv2.dilate(d, element)
        result = d

        res = result.copy()
        contours, hier = cv2.findContours(res, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.current_rect_count = len(contours)
        self.rects = self.choose_contour(contours)
        if len(self.rects) > 0:
            self.current_biggest_rect_area = area(self.rects[0])
        else:
            self.current_biggest_rect_area = 0
        middle_roi = get_roi(result, self.middle_rect)
        self.current_middle_non_zero = cv2.countNonZero(middle_roi)

        self.feedback()

        # DEBUG
        # draw_rects(result, self.rects, color=(255,255,255))
        # draw_rects(result, [self.middle_rect], color=(255,255,255))
        # cv2.imshow('Result', result)


    def calibrate_threshold(self, img):
        hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h1,s1,v1 = cv2.split(hsv1)
        dummy, v1 = cv2.threshold(v1, self.threshold, 255, cv2.THRESH_BINARY)
        v1 = self.smart_filter(h1, v1)
        contours, hier = cv2.findContours(v1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = [cnt for cnt in contours if cv2.contourArea(cnt) > 3000]
        c1 = self.biggest_cnt(cnts)
        if c1 == None:
            self.feedback()
            return None
        c1 = cv2.approxPolyDP(c1, 5, True)
        self.rects = [cv2.boundingRect(c1)]
        v1 = np.zeros(v1.shape, np.uint8)
        cv2.drawContours(v1,[c1],-1,(255,0,0),-1)
        hull = cv2.convexHull(c1, returnPoints = False)
        defects = cv2.convexityDefects(c1, hull)
        sum_area = 0
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i][0]
            start = tuple(c1[s][0])
            end = tuple(c1[e][0])
            far = tuple(c1[f][0])
            t = np.array([[start], [end], [far]])
            sum_area += cv2.contourArea(t)
        self.defects_count = defects.shape[0]
        self.avg_area = cv2.contourArea(c1)
        self.prc_avg_area = self.avg_area/float(area(self.rects[0]))
        self.prc_defect_area = (sum_area/defects.shape[0])/self.avg_area

        self.feedback()

        # DEBUG
        # draw_rects(v1, self.rects, color=(255,255,255))
        # cv2.imshow('threshold', v1)

    def feedback(self):
        if self.phase == 0:
            if len(self.rects) == 0 or self.stabilize < 4:
                self.stabilize += 1
                return 
            self.enviroment_info.append(EnviromentInfo(self.current_middle_non_zero, 
                                                       self.current_rect_count, 
                                                       self.current_biggest_rect_area))
            self.counter += 1
            self.stabilize = 0
            if self.counter < len(self.conf_to_check):
                self.current_conf = self.conf_to_check[self.counter]
            else:
                self.phase += 1
        elif self.phase == 1:
            self.choose_best_configuration()
            self.phase += 1
        elif self.phase == 2:
            self.choose_time_of_day()
            if self.light != "Day":
                self.phase += 1
            else:
                self.end = 1
        elif self.phase == 3:
            if self.stabilize < 4:
                self.stabilize += 1
                return
            self.stabilize = 0
            # print "****", self.threshold, self.avg_area, self.prc_defect_area
            self.threshold_info.append(ThresholdInfo(self.threshold,
                                                     self.defects_count,
                                                     self.prc_defect_area,
                                                     self.avg_area,
                                                     self.prc_avg_area))
            self.threshold += 10
            if self.threshold > self.threshold_end:
                self.phase += 1
        elif self.phase == 4:
            repeat = self.choose_best_threshold()
            if repeat:
                self.threshold_start -= 10
                self.threshold_end -= 15
                self.threshold = self.threshold_start
                self.phase -= 1
                self.threshold_info = []
                if self.threshold_end <= self.threshold_start:
                    self.end = 1
            else:
                self.end = 1

    def choose_time_of_day(self):
        indicator = self.conf_match[self.choice]
        if self.enviroment_info[self.choice].rect_count > 100 and indicator == "Day":
            indicator = "Night"
        self.light = indicator

    def choose_best_threshold(self):
        if self.threshold_info[0].avg_area < 12000:
            #print "Lower boundary, area too small:", self.threshold_info[0].avg_area
            return True
        for i in range(0, len(self.threshold_info)):
            # print self.threshold_info[i].threshold
            # print self.threshold_info[i].avg_area
            # print self.threshold_info[i].prc_defect_area
            # print self.threshold_info[i].prc_avg_area
            # print "********"
            if 0.008 < self.threshold_info[i].prc_defect_area < 0.025 and self.threshold_info[i].prc_avg_area > 0.71 and \
               self.threshold_info > 14000:
                self.final_threshold = self.threshold_info[i].threshold
                return False
        return True

    def choose_best_configuration(self):
        best_area = -1
        tmp_area = 0
        best_count = -1
        tmp_count = 99999
        best_non_zero = -1
        tmp_non_zero = 0
        for i, e in enumerate(self.enviroment_info):
            if e.middle_non_zero > tmp_non_zero:
                tmp_non_zero = e.middle_non_zero
                best_non_zero = i
            if e.rect_count < tmp_count:
                tmp_count = e.rect_count
                best_count = i
            if e.biggest_rect_area > tmp_area:
                tmp_area = e.biggest_rect_area
                best_area = i
            # print i, self.conf_to_check[i], e.middle_non_zero, e.rect_count, e.biggest_rect_area, "*"
        l = len(self.enviroment_info)
        pts = [(self.enviroment_info[b].points(), b) for b in range(0, l)]
        self.choice = sorted(pts)[-1][1]
        # print sorted(pts)
        self.best_conf = self.conf_to_check[self.choice]


if __name__ == "__main__":
    c = cv2.VideoCapture(0)
    clbr = Calibration()
    while(1):    
        _,f = c.read()
        clbr.update(f)
        #clbr.show_mode(f)
        k = cv2.waitKey(20)	
        if clbr.end:
            print clbr.best_conf, "final"
            print clbr.final_threshold
            break
        if k == 27:
            break
    cv2.destroyAllWindows()
    c.release()