import cv2
import numpy as np
import time
from main_utils import draw_rects, \
    get_roi

absd = cv2.absdiff


def get_longest_ranges(conf):
    """
        Pick two longest ranges of HSV values from
        given ranges. If there's range with 0 always
        pick it.
    """
    add_zero = False
    zero_range = [0, 0]
    if conf[0] == 0:
        add_zero = True
        zero_range[1] = conf[1]

    indexes = range(0, len(conf) - 1, 2)
    best_ranges = [(conf[ind+1] - conf[ind], [conf[ind], conf[ind+1]]) for ind in indexes]
    best_ranges = sorted(best_ranges)
    final = best_ranges[-1][1] + best_ranges[-2][1]
    if best_ranges[-1][1][0] > best_ranges[-2][1][0]:
        final = best_ranges[-2][1] + best_ranges[-1][1]

    if add_zero:
        if final[0] != 0:
            final = zero_range + final[0:2]
        else:
            final = zero_range + final[2:4]
    return final

def get_ranges(hist, threshold):
    """
        From histogram of HSV values calculates ranges to use
        in detecting human skin.
    """
    rngs = []
    if len(hist) == 0:
        return [255, 255, 255, 255]

    for i in range(0, len(hist)):
        if hist[i] > threshold:
            rngs.append(i)
    #print rngs, "high values"
    if len(rngs) == 0:
        return [255, 255, 255, 255]
    elif len(rngs) == 1:
        return [rngs[0], rngs[0], 255, 255]

    current = rngs[0]
    values = [rngs[0]]
    for i in range(0, len(rngs)):
        if (rngs[i] <= current + 3) or \
           (rngs[i] > 120 and rngs[i] <= current + 6):
            current = rngs[i]
        else:
            values.append(current)
            values.append(rngs[i])
            current = rngs[i]
    if current == rngs[-1]:
        values.append(current)
    if len(values) == 2:
        values.extend([255, 255])
    if len(values) > 4:
        values = get_longest_ranges(values)
    
    return values


def clean_conf_h(conf):
    """
        Changes ranges of detected HSV values with respect to 
        prior knowledge of HSV ranges for human skin in 
        artificial and natural light conditions.
    """
    if conf[2] == 255 and conf[3] == 255 and conf[0] < 15 and conf[1] > 20:
        if conf[1] >= 30:            
            conf[0] = 0
            conf[1] = 20
            conf[2] = 30
            conf[3] = 30
        elif conf[1] >= 25:
            c0 = conf[0]
            conf[0] = 0
            conf[1] = 0
            conf[2] = c0
            conf[3] = 25

    if conf[1] <= 18 and conf[0] < 12:
        conf[0] = 0

    # forbidden ranges
    if (35 < conf[2] <= 50) and (35 < conf[3] <= 50):
        conf[2] = 35
        conf[3] = 35 
    if 50 < conf[2] < 120:
        conf[2] = 120
    if 50 < conf[3] < 120:
        conf[3] = 120
    return conf

def clear_conf_v(conf):
    if conf[1] - conf[0] > 30:
        conf[0] -= 5
        conf[1] -= 10
    elif conf[1] - conf[0] > 25:
        conf[0] -= 3
        conf[1] -= 6
    return conf

class Calibration2(object):

    def __init__(self, height=480, width=640):
        self.element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        self.h = height
        self.w = width
        self.rect = [0, 0, 0, 0]

        self.conf_h = [0, 0, 0, 0]
        self.conf_yv = [0, 0, 0, 0]
        self.thr = 90
        self.light = "Day"

        self.last = np.zeros((height, width), np.uint8)
        self.end = 0
        self.cnt = 0
        self.cnt_max = 30

        self.yv_remove_threshold = 0.04
        self.h_remove_threshold = 0.03

    def biggest_cnt(self, cnts):
        biggest = None
        biggest_area = 0
        for cnt in cnts:
            m = cv2.moments(cnt)
            rect = cv2.boundingRect(cnt)
            if m["m00"] > biggest_area and rect[1] < self.h/2:
                biggest = cnt
                biggest_area = m["m00"]
        return biggest

    def get_head_rect(self, img, cnt):
        rect = list(cv2.boundingRect(cnt))
        if rect[3] > self.h/3:
            rect[3] = rect[3]/2
        approx_roi = get_roi(img, rect)
        roi = approx_roi.copy()
        #cv2.imshow('roihead', roi)
        cnts, hier = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = self.biggest_cnt(cnts)

        rect_inside = list(cv2.boundingRect(cnt))
        rect[0] = rect[0] + rect_inside[0]
        rect[1] = rect[1] + rect_inside[1]
        rect[2] = rect_inside[2]
        rect[3] = rect_inside[3]
        return rect

    def discover_light(self, value_img):
        dummy, value_240 = cv2.threshold(value_img, 240, 255, cv2.THRESH_BINARY)
        dummy, value_thr = cv2.threshold(value_img, self.thr, 255, cv2.THRESH_BINARY)
        no_white_240 = cv2.countNonZero(value_240)
        no_white_thr = cv2.countNonZero(value_thr)
        cnts, hier = cv2.findContours(value_240, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        rects = [cv2.boundingRect(cnt) for cnt in cnts if cv2.contourArea(cnt) > 20000]
        if no_white_thr*0.4 <= no_white_240 and len(rects) > 0:
            self.light = "Day"
            self.thr = 240
        else:
            self.light = "Night"

    def update(self, img):
        hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        h1,s1,v1 = cv2.split(hsv1)
        value_img = v1.copy()
        y, u, v = cv2.split(yuv)
        self.frame = h1
        orig = h1.copy()
        if self.thr < 240:
            self.thr, v1 = cv2.threshold(v1, 0, 255, cv2.THRESH_OTSU)
        else:
            dummy, v1 = cv2.threshold(v1, 240, 255, cv2.THRESH_BINARY)
        v2 = v1.copy()
        cnts, hier = cv2.findContours(v2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = self.biggest_cnt(cnts)
        if cnt is not None:
            self.rect = self.get_head_rect(v1, cnt)
            #draw_rects(v1, [self.rect], color=(100,0,0))
            #cv2.imshow('thr', v1)
            roi_h = get_roi(orig, self.rect)
            roi_v = get_roi(v, self.rect)
            mask = get_roi(v1, self.rect)

            hist = cv2.calcHist([roi_h], [0], mask, [256], [0,256])
            hist = np.array([i/hist.max() for i in hist])
            conf = get_ranges(hist, threshold=self.h_remove_threshold)
            if conf[1] - conf[0] > 30:
                self.h_remove_threshold += 0.02
            self.conf_h = clean_conf_h(conf)

            hist = cv2.calcHist([roi_v], [0], mask, [256], [0,256])
            hist = np.array([i/hist.max() for i in hist])
            self.conf_yv = get_ranges(hist, threshold=self.yv_remove_threshold)
            if self.conf_yv[1] - self.conf_yv[0] > 30:
                self.yv_remove_threshold += 0.01
            self.conf_yv = clear_conf_v(self.conf_yv)

            self.discover_light(value_img)
            self.cnt += 1
        if self.cnt >= self.cnt_max:
            self.end = 1
            print "\n"


def test_main():
    c = cv2.VideoCapture(0)
    LIGHT = "Night"
    CFG_HSV = [0, 0, 0, 0]
    CFG_THR = 90

    def hsv(img):
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h,s,v = cv2.split(img_hsv)
        color_range = CFG_HSV
        d = cv2.inRange(h, np.array(color_range[0],np.uint8), 
                           np.array(color_range[1],np.uint8))
        d2 = cv2.inRange(h, np.array(color_range[2],np.uint8), 
                            np.array(color_range[3],np.uint8))
        d = cv2.bitwise_or(d, d2)
        return d
    def u__(img, color_range):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        d = cv2.inRange(u, np.array(color_range[0],np.uint8), 
                           np.array(color_range[1],np.uint8))
        d2 = cv2.inRange(u, np.array(color_range[2],np.uint8), 
                            np.array(color_range[3],np.uint8))
        d = cv2.bitwise_or(d, d2)
        return d
    def v__(img, color_range):
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y, u, v = cv2.split(img_yuv)
        d = cv2.inRange(v, np.array(color_range[0],np.uint8), 
                           np.array(color_range[1],np.uint8))
        d2 = cv2.inRange(v, np.array(color_range[2],np.uint8), 
                            np.array(color_range[3],np.uint8))
        d = cv2.bitwise_or(d, d2)
        return d
    clbr = Calibration2()
    cnt = 0
    while (not clbr.end):
        _,f = c.read()
        clbr.update(f)
        k = cv2.waitKey(20)
        if k == 27:
            break
        cnt += 1
        if cnt > 120:
            break
    print clbr.conf_h, clbr.conf_yv, clbr.thr, clbr.light
    LIGHT = clbr.light
    CFG_HSV = clbr.conf_h
    #CFG_HSV = [0,22, 23, 23]
    CFG_THR = clbr.thr
    #clbr.conf_yv = [128, 133, 221, 255]
    while True:
        _,f = c.read()
        img = hsv(f)
        imgv = v__(f, clbr.conf_yv)
        cv2.imshow('H', img)
        cv2.imshow('V', imgv)   
        d = cv2.bitwise_and(img, imgv)
        cv2.imshow('D', d)
        k = cv2.waitKey(20)
        if k == 27:
            break

    cv2.destroyAllWindows()
    c.release()


#test_main()