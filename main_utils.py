"""
    Different utility functions for hand gesture detection alogorithm
"""

import cv2

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
        
def draw_rects(image, rects, wide=1):
    """ draws rects in the image """
    for rect in rects:
        x,y,w,h = rect
        cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0), wide)
        
def draw_circles(self, img, positions):
    r = 22
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
    
def is_far_away(rect1, rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2
    far_away_x = min(int(1.5*w1), CFG_FAR_AWAY_X)
    far_away_y = min(2*h1, CFG_FAR_AWAY_Y)
    if abs(x2 - x1) >= far_away_x or abs(y2 - y1) >= far_away_y:
        return True
    return False
    
def is_big_enough(rect1):
    x1,y1,w1,h1 = rect1
    if w1*h1 >= 9600:
        return True
    return False
    
    
def one_inside_another(last_rect, before_last_rect):
    left = last_rect[0] >= before_last_rect[0]-5
    right = last_rect[0] + last_rect[2] <= before_last_rect[0] + before_last_rect[2]+5
    up = last_rect[1] >= before_last_rect[1]-5
    down = last_rect[1] + last_rect[3] <= before_last_rect[1] + before_last_rect[3]+5
    sum = int(left) + int(right) + int(up) + int(down)
    if sum >= 3:
        area_last = last_rect[2]*last_rect[3]
        area_before_last = before_last_rect[2]*before_last_rect[3]
        if area_before_last > 3*area_last:
            return True
    return False
##
# alogrytm odrozniania reki od twarzy
# 1. bgr2gray
# 2. medianBlur, erode, dilate
# 3. przejechac filterm 10na10
# 4. wczytac obrazy biale/nie biale do tablic w formie 1,0
# np. jesli dany fragment obrazu 10na10 jest bialy to w tablicy odpowiada to jedynce
# 5. wyszukac zamkniete lub prawie zamkniete cykle 1 w tablicy

# alternatywnie mozna zaimplementowac jakas funkcje czyszczaca:
# jedziemy od rogu (0,0) obrazu -> jesli gorny i prawy sasiad sa biali to ten nowy tez