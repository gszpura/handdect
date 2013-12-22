import cv2
import numpy as np
import time
from main_utils import draw_rects

absd = cv2.absdiff

def _height(p):
    return p[0][1]

def get_roi(img, rect):
    x,y,w,h = rect
    roi = img[y:y+h, x:x+w]
    return roi


def get_ranges(hist):
    rngs = []
    if len(hist) == 0:
        return [255, 255, 255, 255]

    low = min(1, len(hist))
    for i in range(low, len(hist)):
        if hist[i] > 0.007:
            rngs.append(i)
    # print rngs
    if len(rngs) == 0:
        return [255, 255, 255, 255]
    elif len(rngs) == 1:
        return [rngs[0], rngs[0], 255, 255]

    current = rngs[0]
    values = [rngs[0]]
    cnt = 0
    for i in range(1, len(rngs)):
        if (rngs[i] <= current + 2) or \
           (rngs[i] > 120 and rngs[i] <= current + 6):
            current = rngs[i]
            cnt += 1
        else:
            if cnt == 0:
                values.append(rngs[i])
                cnt = 0
                continue
            values.append(current)
            values.append(rngs[i])
            current = rngs[i]
            cnt = 0
    if current == rngs[-1]:
        values.append(current)
    if len(values) > 4:
        values = values[:4]
    if len(values) == 2:
        values.extend([255, 255])
    return values


def clean_conf(conf):
    # near zero changes
    if conf[0] < 3 and conf[1] <= 15:
        conf[0] = 1
        if conf[1] > 3:
            conf[1] = 2
    if conf[1] > 35:
        conf[1] = 35

    # daylight changes
    if 36 < conf[3] < 80:
        conf[3] = 36 
    if 36 < conf[2] < 80:
        conf[2] = 36

    # night changes
    if 150 < conf[2] < 190:
        conf[2] = 150
    if 80 < conf[2] < 120:
        conf[2] = 120
    if 80 < conf[3] < 190:
        conf[3] = 190
    if 200 < conf[3] < 255:
        conf[3] = 200
    return conf

class Calibration2(object):

    def __init__(self, height=480, width=640):
        self.element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        self.h = height
        self.w = width
        self.rect = [0, 0, 0, 0]

        self.best_conf = [1, 2, 3, 4]
        self.thr = 90
        self.light = "Day" #or "Night" or "DayDim"

        self.last = np.zeros((height, width), np.uint8)
        self.end = 0
        self.cnt = 0
        self.cnt_max = 20

    def biggest_cnt(self, cnts):
        biggest = None
        biggest_area = 0
        for cnt in cnts:
            m = cv2.moments(cnt)
            if m["m00"] > biggest_area:
                biggest = cnt
                biggest_area = m["m00"]
        return biggest

    def cut_the_crap(self, cnt):
        rect = list(cv2.boundingRect(cnt))
        if rect[3] > self.h/3:
            rect[3] = rect[3]/2
        y2 = rect[3] + rect[1]
        cnt2 = sorted(cnt, key=_height)
        cnt2 = np.array([p for p in cnt2 if p[0][1] < y2])
        return cnt2

    def discover_light(self):
        conf = self.best_conf
        if conf[0] < 3 and conf[1] < 3:
            self.light = "Night"
        elif conf[1] <= 30 and conf[3] <= 30:
            self.light = "DayDim"
        else:
            self.light = "Day"

    def update(self, img):
        hsv1 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h1,s1,v1 = cv2.split(hsv1)
        orig = h1.copy()
        self.thr, v1 = cv2.threshold(v1, 0, 255, cv2.THRESH_OTSU)
        v2 = v1.copy()
        cnts, hier = cv2.findContours(v2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = self.biggest_cnt(cnts)
        if cnt is not None:
            cnt = self.cut_the_crap(cnt)
            self.rect = list(cv2.boundingRect(cnt))
            draw_rects(v1, [self.rect], color=(100,0,0))
            # cv2.imshow('otsu', v1)
            roi = get_roi(orig, self.rect)
            mask = get_roi(v1, self.rect)
            hist = cv2.calcHist([roi], [0], mask, [256], [0,256])
            hist = np.array([i/hist.max() for i in hist])
            conf = get_ranges(hist)
            print "*",
            self.best_conf = clean_conf(conf)
            self.discover_light()
            self.cnt += 1
        if self.cnt >= self.cnt_max:
            self.end = 1
            print "\n"


def test_main():
    c = cv2.VideoCapture(0)
    LIGHT = "Night"
    CFG_HSV = [1, 2, 145, 190]
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
    print "Calibration2"
    clbr = Calibration2()
    cnt = 0
    while (not clbr.end):
        _,f = c.read()
        clbr.update(f)
        k = cv2.waitKey(20)
        if k == 27:
            break
        cnt += 1
        if cnt > 20:
            break
    print clbr.best_conf
    print clbr.thr
    print clbr.light
    LIGHT = "Night" #clbr.light
    CFG_HSV = clbr.best_conf #[1,30,120,190] 
    CFG_THR = clbr.thr

    while True:
        _,f = c.read()
        img = hsv(f)
        cv2.imshow('res', img)
        k = cv2.waitKey(20)
        if k == 27:
            break

    cv2.destroyAllWindows()
    c.release()


#test_main()