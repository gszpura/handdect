"""
    Different utility functions for hand gesture detection alogorithm
"""

import cv2
from math import sqrt
from config import HEIGHT, WIDTH

LATTICE_X = 20
LATTICE_Y = 20

CFG_WIDTH = HEIGHT
CFG_HEIGHT = WIDTH

#for entering a frame/window
CFG_WIDTH_RATIO = 0.75
CFG_HEIGHT_RATIO = 0.75

#for box jump
CFG_DX = CFG_WIDTH/6
CFG_DY = CFG_HEIGHT/4
CFG_DW = CFG_WIDTH/10
CFG_DH = CFG_HEIGHT/3

CFG_FAR_AWAY_X = int(CFG_WIDTH/3.5)
CFG_FAR_AWAY_Y = int(CFG_HEIGHT/2.5)

MINIMUM_W = int(0.1*WIDTH)
MINIMUM_H = int(0.14*HEIGHT)


def draw_boxes(image, boxes, wide=1):
    """ draws boxes in the image """
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), wide)

        
def draw_rects(image, rects, wide=1, color=(255,0,0)):
    """ draws rects in the image """
    for rect in rects:
        x,y,w,h = rect
        cv2.rectangle(image, (x, y), (x+w, y+h), color, wide)

        
def draw_circles(img, positions, r=20):
    for pos in positions:
        cv2.circle(img, pos, r, (255,0,0))

def draw_info(img, text, x, y):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (x, y), font, 1, (255,255,255), 2)
        
        
def close_to_edge(rect):
    x,y,w,h = rect
    big_x = CFG_WIDTH_RATIO*CFG_WIDTH
    big_y = CFG_HEIGHT_RATIO*CFG_HEIGHT
    if x < CFG_WIDTH - big_x:
        return True
    if x + w > big_x or y > big_y:
       return True
    return False
    
    
def close_to_each_other(rect1, rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2
    if abs(x2 - x1) <= CFG_DX and abs(y2 - y1) <= CFG_DY:
        #if abs(w2 - w1) <= CFG_DW and abs(h2 - h1) <= CFG_DH:
        return True
    return False

    
def close_to_each_other_central(rect1, rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2
    if abs( (x2+w2/2) - (x1+w1/2) ) <= CFG_DX and abs( (y2+h2/2) - (y1+h1/2)) <= CFG_DY:
        return True
    return False


def is_far_away(rect1, rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2
    far_away_x = min(int(1.2*w1), CFG_FAR_AWAY_X)
    far_away_y = min(2*h1, CFG_FAR_AWAY_Y)
    if abs(x2 - x1) >= far_away_x or abs(y2 - y1) >= far_away_y:
        return True
    return False


def is_very_close(rect, stable_rect, dm=20):
    x,y,w,h = rect
    xh,yh,wh,hh = stable_rect
    if xh - dm < x < xh + dm and \
        yh - dm < y < yh + dm and \
        xh + wh - dm < x + w < xh + wh + dm and \
        yh + hh - dm < y + h < yh + hh + dm:
            return True
    return False


def combine_rects(last, to_merge):
    x1,y1,w1,h1 = last
    x2,y2,w2,h2 = to_merge
    x = int(0.6*x1 + 0.4*x2)
    y = int(0.6*y1 + 0.4*y2)
    w = int(0.7*w1 + 0.3*w2)
    h = int(0.7*h1 + 0.3*h2)
    return (x1, y1, max(70, w), max(80, h))


def average_rect(rect1, rect2):
    x = (4*rect1[0] + rect2[0])/5
    y = (4*rect1[1] + rect2[1])/5
    w = (4*rect1[2] + rect2[2])/5
    h = (4*rect1[3] + rect2[3])/5
    return [x,y,w,h]


def average_from_rects(rects):
    x = 0; y = 0; w = 0; h = 0
    for rect in rects:
        x += rect[0]
        y += rect[1]
        w += rect[2]
        h += rect[3]
    ln = len(rects)
    if ln > 0:
        return [x/ln, y/ln, w/ln, h/ln]
    else:
        return [300, 300, 0, 0]


def is_near_rect(reference, rect):
    if reference is None:
        return False
    x,y,w,h = reference
    xr, yr, wr, hr = rect
    cx = xr + wr/2
    cy = yr + hr/2
    ext = 50
    if cx > x and cx < x + w and cy > y - 20  and cy < y + w + ext:
        return True
    return False


def distance_between_rects(rect1, rect2):
    cx1 = rect1[0] + rect1[2]/2
    cy1 = rect1[1] + rect1[3]/2

    cx2 = rect2[0] + rect2[2]/2
    cy2 = rect2[1] + rect2[3]/2
    distance = sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    return distance


def one_inside_another(current, previous, ratio=3, rigid=False):
    """
        Checks if rect 'current' is inside rect 'previous'.
        @param ratio: ratio between areas which must be kept
            to say that current rect is inside previous one
    """
    if current is None or previous is None:
        return False
    left = current[0] >= previous[0]-5
    right = current[0] + current[2] <= previous[0] + previous[2]+5
    up = current[1] >= previous[1]-5
    down = current[1] + current[3] <= previous[1] + previous[3]+5
    _sum = int(left) + int(right) + int(up) + int(down)
    if rigid and _sum == 4:
        return True
    if _sum >= 3:
        current_area = current[2]*current[3]
        previous_area = previous[2]*previous[3]
        if previous_area > ratio*current_area:
            return True
    return False


def is_real_check(roi):
    if roi is None:
        return False
    w, h = roi.shape
    all_pixels = 20*w
    part_of_roi = roi[h/4:3*h/4,:]
    amount = cv2.countNonZero(part_of_roi)
    if amount > 0.05*all_pixels:
        return True
    else:
        return False

def get_roi(img, rect):
    x,y,w,h = rect
    roi = img[y:y+h, x:x+w]
    return roi


def cnt_area(cnt):
    """Function returns area for contour"""
    return cv2.moments(cnt)["m00"]


def rev_cnt_area(cnt):
    area = cv2.moments(cnt)["m00"]
    if area == 0:
        return 1
    return 1/area


def get_biggest_cnt(cnts, how_many=1):
    if how_many == 1:
        try:
            biggest = [max(cnts, key=cnt_area)]
        except Exception as e:
            print e, "get_biggest_cnt"
            biggest = None
    else:
        biggest = []
        sort = sorted(cnts, key=rev_cnt_area)
        biggest = sort[0:how_many]
        if len(biggest) == 0:
            return None
    return biggest


def minimal_rect(rect):
    """
    If rect is too small make it a bit bigger
    in ordert to not to miss part of the hand.
    """
    x, y, w, h = rect
    if w < MINIMUM_W:
        x = max(0, x - (MINIMUM_W - 10))
        w = min(640, w + MINIMUM_W)
    if h < MINIMUM_H:
        y = max(0, y - (MINIMUM_H - 10))
        h = min(480, h + MINIMUM_H)
    if h < MINIMUM_H*2 and w > MINIMUM_W*2:
        y = max(0, y - MINIMUM_H)
        h = min(480, h + MINIMUM_H)
    return [x, y, w, h]


def find_contours(roi):
    contours, hierarchy = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def fill_in_contour(roi, cnt):
    cv2.drawContours(roi, [cnt], -1, (255,0,0), -1)
