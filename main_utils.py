"""
    Different utility functions for hand gesture detection alogorithm
"""

import cv2
from math import sqrt

LATTICE_X = 20
LATTICE_Y = 20

CFG_WIDTH = 640
CFG_HEIGHT = 480

#for entering a frame/window
CFG_WIDTH_RATIO = 0.85
CFG_HEIGHT_RATIO = 0.8

#for box jump
CFG_DX = CFG_WIDTH/10
CFG_DY = CFG_HEIGHT/4
CFG_DW = CFG_WIDTH/10
CFG_DH = CFG_HEIGHT/3

CFG_FAR_AWAY_X = int(CFG_WIDTH/3.5)
CFG_FAR_AWAY_Y = CFG_HEIGHT/2


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
        

        
def close_to_edge(rect):
    x,y,w,h = rect
    big_x = CFG_WIDTH_RATIO*CFG_WIDTH
    big_y = CFG_HEIGHT_RATIO*CFG_HEIGHT
    if x < CFG_WIDTH - big_x:
        return True
    if x > big_x or y > big_y:
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
    
def not_close(rect1, rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2
    if x1 + w1 < x2 or x1 > x2 + w2 or y1 + h1 < y2 or y1 > y2 + h2:
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


def average_queue(queue):
    x = 0
    y = 0
    w = 0
    h = 0
    for rect in queue:
        x += rect[0]
        y += rect[1]
        w += rect[2]
        h += rect[3]
    l = len(queue)
    return [x/l, y/l, w/l, h/l]

def is_near_rect(reference, rect):
    x,y,w,h = reference
    xr, yr, wr, hr = rect
    cx = xr + wr/2
    cy = yr + hr/2
    ext = 150
    if cx > x and cx < x + w and cy > y - 20  and cy < y + w + ext:
        return True
    return False

def correct_rect(rect):
    xt,yt,wt,ht = rect
    rect = list(rect)
    if wt*ht < 4000:
        r0 = (rect[0] - 40) > 0 and (rect[0] - 40) or 0
        r1 = (rect[1] - 40) > 0 and (rect[1] - 40) or 0
        rect[0] = r0
        rect[1] = r1
        rect[2] = rect[2] + 80
        rect[3] = rect[3] + 80
    elif wt*ht < 13000:
        rect[3] += 50
        r0 = (rect[0] - 30) > 0 and (rect[0] - 30) or 0
        rect[0] = 0
        rect[2] = rect[2] + 60
    if rect[2] > 1.25*rect[3]:
        rect[3] = rect[3] + 60
    if rect[0] < 0:
        rect[0] = 0
    if rect[1] < 0:
        rect[1] = 0
    return tuple(rect)

def distance_between_rects(rect1, rect2):
    cx1 = rect1[0] + rect1[2]/2
    cy1 = rect1[1] + rect1[3]/2

    cx2 = rect2[0] + rect2[2]/2
    cy2 = rect2[1] + rect2[3]/2
    distance = sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
    return distance

def further_from_rect(rectx, rect1, rect2):
    d1 = distance_between_rects(rectx, rect1)
    d2 = distance_between_rects(rectx, rect2)
    print d2, d1, "DDD"
    print rect2, rect1
    print rectx
    if d2 > d1:
        return rect2
    return rect1


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
