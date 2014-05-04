"""
    Trackery:
        -> TrackerAL for HAAR Cascades
        -> StateTracker for background substraction + HSV + YUV

    TODO pracy dyplomowej:
    - convectivyDefects: jestli brak wiekszych w FACE to oznacza, ze to PALM 
    - przepisac do cythona
    - HAAR
    - porownania i testy

    TODO trackera:
    - przechodzenie reki nad glowa: 
        przeskakiwanie boxa: dodac 3 rect i sprawdzac czy nie jest blisko last_rect a rect[n] nie jest blisko head_rect
        
    - zamiana dloni i glowy 

    TODO aplikacji:
    - wykorzystac analize rectow w kazdej klatce

    FUTURE TASKS/KNOWN BUGS:
    - kalibracja tylko na podstawie glowy lub kalibracja na podstawie uniesionej dloni

"""


import cv2
import numpy as np
from copy import copy
from collections import deque

from main_utils import draw_circles, \
    draw_rects, \
    close_to_edge, \
    close_to_each_other, \
    close_to_each_other_central, \
    is_far_away, \
    is_very_close, \
    combine_rects, \
    one_inside_another, \
    CFG_HEIGHT, CFG_WIDTH, \
    draw_circles, \
    average_rect, \
    is_near_rect, \
    distance_between_rects, \
    further_from_rect, \
    is_real_check, \
    get_roi, \
    average_from_rects
from shape_discovery import ShapeDiscovery


class RectSaver(object):

    def __init__(self):
        self.img1 = None
        self.img2 = None
        self.state = 0

    def save_hand(self, img, rect1=None, rect2=None):
        if rect1:
            x,y,w,h = rect1
            img1 = img[y:y+h, x:x+w] 
            self.save_img(img1)
        if rect2:
            x,y,w,h = rect2
            img2 = img[y:y+h, x:x+w]
            self.save_img(img2)

    def save_img(self, img):
        cv2.imwrite('C:\\Python27\\pdym\\imgs\img%s.png' % self.state, img)
        self.state += 1

    @staticmethod
    def show_hand(img, rect1=None, rect2=None):
        if rect1:
            x,y,w,h = rect1
            zero = np.zeros((300,300), np.unit8)
            img1 = img[y:y+h, x:x+w]
            zero[0:h, 0:w] = img1 
            cv2.imshow('1', zero)
        if rect2:
            x,y,w,h = rect2
            zero = np.zeros((300,300), np.unit8)
            img2 = img[y:y+h, x:x+w]
            zero[0:h, 0:w] = img2
            cv2.imshow('2', zero)    
            

class TrackerAL:

    def __init__(self):
        self.x = -1
        self.y = -1
        self.vx = 0
        self.vy = 0
        self.predicted_x = -1
        self.predicted_y = -1
        self.sizex_up = 60
        self.sizex_down = 70
        self.sizey_up = 130
        self.sizey_down = 30
        self.the_group = [-1,-1,-1,-1]
        self.enter_counter = 0
        self.jump_counter = 0
        self.rects = None
            
    def draw_bounding_boxes(self, img, rects):
        for rec in rects:
            x,y,w,h = rec
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 1)
            
    def draw_groups(self, img, groups):
        for g in groups:
            x,x2,y,y2 = g
            cv2.rectangle(img, (x,y),(x2,y2), (255,0,0), 1)
    
    def draw_single_group(self, img, group):
        x,x2,y,y2 = group
        cv2.rectangle(img, (x,y),(x2,y2), (255,0,0), 3)
            
        
    def process_bounding_boxes(self, rects):
        sizex_up = self.sizex_up
        sizex_down = self.sizex_down
        sizey_up = self.sizey_up
        sizey_down = self.sizey_down
        rects2 = sorted(rects)

        ret = []
        groups = []
        change = False
        change_cnt = 0
        for rect in rects2:
            cx = rect[0] + rect[2]/2
            cy = rect[1] + rect[3]/2
            if len(groups) == 0:    
                search_area = [cx-sizex_down, cx+sizex_up, cy-sizey_down, cy+sizey_up]
                group = [rect[0], rect[0] + rect[2], rect[1], rect[1] + rect[3]] #x1,x2,y1,y2
                groups.append([search_area, group])
                continue
            change_cnt = 0
            for g in groups:
                sa, gr = g
                if sa[0] < cx < sa[1] and sa[2] < cy < sa[3]:
                    change = True
                    break
                change_cnt += 1
            if change:
                sa = groups[change_cnt][0]
                gr = groups[change_cnt][1]
                
                x2 = sa[1]
                y1 = sa[2]
                y2 = sa[3]
                
                x2gr = gr[1]
                y1gr = gr[2]
                y2gr = gr[3]
                #sa
                if cx > x2 - sizex_up:  
                    groups[change_cnt][0][1] = cx + sizex_up
                if cy < y1 + sizey_down:
                    groups[change_cnt][0][2] = cy - sizey_down
                elif cy > y2 - sizey_up:
                    groups[change_cnt][0][3] = cy + sizey_up
                #gr    
                if rect[0] + rect[2] > x2gr:
                    groups[change_cnt][1][1] = rect[0] + rect[2]
                if rect[1] < y1gr:
                    groups[change_cnt][1][2] = rect[1]
                if rect[1] + rect[3] > y2gr:
                    groups[change_cnt][1][3] = rect[1] + rect[3]
            elif len(groups) < 3:
                search_area = [cx-sizex_down, cx+sizex_up, cy-sizey_down, cy+sizey_up]
                group = [rect[0], rect[0] + rect[2], rect[1], rect[1] + rect[3]]
                groups.append([search_area, group])
            change = False
        if len(groups) > 0:
            ret = zip(*groups)[0]
        return ret
        
    def get_hsv_limits(self, img):
        cp = img.copy()
        contours, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.rects = [cv2.boundingRect(cnt) for cnt in contours if 100 < cv2.contourArea(cnt) < 20000]
        x1, x2, y1, y2 = [300]*4
        for rec in self.rects:
            x, y, w, h = rec
            if x < x1:
                x1 = x
            if x + w > x2:
                x2 = x + w
            if y < y1:
                y1 = y
            if y + h > y2:
                y2 = y + h
        return (x1, x2, y1, y2)
        
    def get_cascades_limits(self):
        return (80, 600, 80, 480)
        
    def best_group(self, groups):
        the_g = self.the_group
        diffx = 60
        diffy = 60
        jump_limit_x = 150
        jump_limit_y_up = 180
        jump_limit_y_down = 250
        best_match = False
        best_score = jump_limit_x + jump_limit_y_down
        y_jump = False
        for group in groups:
            if the_g[0] == -1:
                cent_x = (group[0]+group[1])/2
                if  cent_x < diffx or (group[2]+group[3])/2 > (480 - diffy) or cent_x > (620 - diffx):
                    self.enter_counter += 1
                    if self.enter_counter > 10:
                        return group
            elif the_g[0] - jump_limit_x < group[0] < the_g[0] + jump_limit_x and the_g[2] - jump_limit_y_down < group[2] < the_g[2] + jump_limit_y_up:
                #elif True:
                score = abs(the_g[0] - group[0]) + abs(the_g[2] - group[2])
                if score < best_score:
                    best_match = group
                    best_score = score
        return best_match
        
        
    def pbb(self, hands):
        """pbb - process bounding boxes"""
        if len(hands) == 0:
            return []
        h = [h.handRect for h in hands]
        h = sorted(h, key=lambda x: x[0])
        xdiff = 28
        ydiff = 135
        second = []
        first = []
        for i,p in enumerate(h):
           if abs(p[0] - h[0][0]) > xdiff or abs(p[1] - h[0][1]) > ydiff:
                second.append(p)
           else:
                first.append(p)    
        #first rectangle
        box = [900, 900, 0, 0]
        for p in first:
            if p[0] < box[0]:
                box[0] = p[0]
            if p[1] < box[1]:
                box[1] = p[1]
            if p[0] + p[2] > box[2]:
                box[2] = p[0] + p[2]
            if p[1] + p[3] > box[3]:
                box[3] = p[1] + p[3]
        box2 = [900, 900, 0, 0]
        for p in second:
            if p[0] < box2[0]:
                box2[0] = p[0]
            if p[1] < box2[1]:
                box2[1] = p[1]
            if p[0] + p[2] > box2[2]:
                box2[2] = p[0] + p[2]
            if p[1] + p[3] > box2[3]:
                box2[3] = p[1] + p[3]
        if len(second) > 0:
            return [box, box2]
        else:
            return [box]
    
    def update(self, img):
        cp = img.copy()
        if not self.rects:
            contours, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            self.rects = [cv2.boundingRect(cnt) for cnt in contours if 100 < cv2.contourArea(cnt) < 20000]
        
        #self.draw_circles(img, positions)
        self.draw_bounding_boxes(img, self.rects)
        groups = self.process_bounding_boxes(self.rects)
        
        #self.draw_groups(img, groups)
        best_group = self.best_group(groups)
        if best_group:
            self.the_group = best_group
        if self.the_group:
            #self.draw_single_group(img, self.the_group)
            pass
        self.rects = None
            
        
##########################################################################################################################################     

def rev_area(rect):
    return 1/float(rect[2]*rect[3])


class StateTracker(object):
    
    def __init__(self):
        self.clear()
        self.out_limit = 10
        self.rsave = RectSaver()
        self.dsc = ShapeDiscovery()
        self.head_rect = [200, 200, 0, 0]
        self.head_biggest = [640, 480, 0, 0]
        self.head_history = deque([], maxlen=20)

    def clear(self):
        self.last_rect = None
        self.before_last_rect = None
        self.really_old_rect = None

        self.predicted_rect = None
        self.prediction_counter = 0
        self.before_prediction_rect = None

        self.gesture = "NONE"

        self.dx = 0
        self.dy = 0
        self.wh = deque([(0, 0)], maxlen=6)
        self.average_wh = (0, 0)
        self.dxy = deque([(0, 0)], maxlen=3)
        self.average_dxy = (0, 0)
        self.rects = []

        self.out_counter = 0
        self.is_not_real_counter = 0
        self.out = True

        self.frame = None
        self.current_types = []
        self.last_roi = None
        self.hand_info = -1

        self.last_resort = [0,0,0,0]

    def update_rects(self, contours):
        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 1500]
        s_rects = sorted(rects, key=rev_area)
        self.rects = s_rects[:2]
        self.rects_small = s_rects[2:4]
        #print self.rects

        
    def update(self, img):
        """things that have to be done before selection of new rect"""
        cp = img.copy()
        contours, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.update_rects(contours)
        self.predicted_rect = self.prediction_calculations()
        self.average_calculations()
        self.really_old_rect = self.before_last_rect
        self.before_last_rect = self.last_rect
        # frames and rois
        if self.last_rect is not None:
            self.last_roi = get_roi(self.frame, self.last_rect)
        else:
            self.last_roi = None
        self.frame = img

    def internal_one_inside_another(self):
        return one_inside_another(self.last_rect, self.before_last_rect, 3, True)
        
    def outside_scope(self):
        if self.last_rect is None:
            return True
        if self.last_rect[0] < 0 or self.last_rect[0] > CFG_WIDTH or self.last_rect[1] < 0 or self.last_rect[1] > CFG_HEIGHT:
            return True
        return False

    def check_borders(self):
        if self.last_rect is None:
            return False
        x,y,w,h = self.last_rect
        cx = self.last_rect[0] + 0.5*self.last_rect[2]
        lx = cx - 0.5*self.average_wh[0]
        rx = self.last_rect[0] + 0.5*self.average_wh[0]
        cy = self.last_rect[1] + 0.5*self.average_wh[1]
        if lx < 0 and self.average_dxy[0] <= -10:
            return True
        if rx > CFG_WIDTH - 40 and self.average_dxy[0] >= 10:
            return True
        if cy > CFG_HEIGHT - 40 and self.average_dxy[1] >= 10:
            return True
        if self.is_not_real_counter > 5:
            return True
        return False
        
    def prediction_calculations(self):
        """
            Finds prediction for next hand bounding rect
            and updates some variables.
        """
        if self.before_last_rect == None:
            self.before_last_rect = self.last_rect
            return self.last_rect
        else:
            if self.internal_one_inside_another():
                return self.before_last_rect
            if self.outside_scope():
                return None

            self.dx = self.last_rect[0] - self.before_last_rect[0]
            self.dy = self.last_rect[1] - self.before_last_rect[1]
            w = (self.last_rect[2] + self.before_last_rect[2])/2
            h = (self.last_rect[3] + self.before_last_rect[3])/2
            x = self.last_rect[0]+self.dx
            y = self.last_rect[1]+self.dy

            self.predicted_rect = [x, y, w, h]
            return self.predicted_rect


    def average_calculations(self):
        """calculates averages from last found rect and earlier rects"""
        if self.internal_one_inside_another():
            self.dxy.append((0, 0))
            self.wh.append((self.last_rect[2], self.last_rect[3]))
            return 
        if self.outside_scope():
            self.dxy.clear()
            self.wh.clear()
            self.dxy.append((0, 0))
            self.wh.append((0, 0))
            return

        self.wh.append((self.last_rect[2], self.last_rect[3]))
        _wh = zip(*self.wh)
        self.average_wh = (sum(_wh[0])/len(_wh[0]), sum(_wh[1])/len(_wh[1]))

        self.dxy.append((self.dx, self.dy))
        _dxy = zip(*self.dxy)
        self.average_dxy = (sum(_dxy[0])/len(_dxy[0]), sum(_dxy[1])/len(_dxy[1]))


    def smooth_random_motions(self, candidate):
        if one_inside_another(candidate, self.last_rect):
            return self.last_rect
        return candidate


    def prediction_limit(self, prediction):
        if prediction:
            self.prediction_counter += 1
        else:
            self.prediction_counter = 0
            self.before_prediction_rect = self.last_rect
        if self.prediction_counter >= 2 and self.last_rect is not None:
            self.last_rect = self.before_prediction_rect

    def shape_analysis(self):
        self.current_types = []
        for rect in self.rects:
            shape_type = self.dsc.discover(self.frame, rect)
            self.current_types.append(shape_type)
            #print BODY_PARTS[val]

    def update_head_biggest(self, rect):
        if rect[0] < self.head_biggest[0]:
            self.head_biggest[0] = rect[0]
        if rect[1] < self.head_biggest[1]:
            self.head_biggest[1] = rect[1]
        if rect[1] + rect[3] > self.head_biggest[1] + self.head_biggest[3]:
            self.head_biggest[3] = rect[1] + rect[3] - self.head_biggest[1]
        if rect[0] + rect[2] > self.head_biggest[0] + self.head_biggest[2]:
            self.head_biggest[2] = rect[0] + rect[2] - self.head_biggest[0]

    def analyze_rects(self, img):
        if self.last_roi is not None:
            is_real = is_real_check(self.last_roi)
        else:
            is_real = False
        if len(self.rects) == 0:
            if is_real:
                self.is_not_real_counter = 0
            else:
                self.is_not_real_counter += 1
        elif len(self.rects) == 1:
            if self.rects[0] == self.last_rect:
                pass
            elif is_real:
                self.is_not_real_counter = 0
            else:
                self.is_not_real_counter += 1
            if not is_near_rect(self.last_rect, self.rects[0]) and \
               (self.current_types[0] in ("FACE", "FACE & HAND", "PALM") or self.is_head(self.rects[0])):
                self.head_history.append(self.rects[0])
                self.update_head_biggest(self.rects[0])
                self.head_rect = average_from_rects(self.head_history)
        elif len(self.rects) == 2:
            if not is_near_rect(self.last_rect, self.rects[1]) and \
               (self.current_types[1] in ("FACE", "FACE & HAND", "PALM") or self.is_head(self.rects[1])):
                self.head_history.append(self.rects[1])
                self.update_head_biggest(self.rects[1])
                self.head_rect = average_from_rects(self.head_history)
            elif not is_near_rect(self.last_rect, self.rects[0]) and \
                 (self.current_types[0] in ("FACE", "FACE & HAND", "PALM") or self.is_head(self.rects[0])):
                self.head_history.append(self.rects[0])
                self.update_head_biggest(self.rects[0])
                self.head_rect = average_from_rects(self.head_history)
            if is_real:
                self.is_not_real_counter = 0
            else:
                self.is_not_real_counter += 1

    def is_head(self, rect):
        if is_very_close(rect, self.head_rect, dm=12):
            return True
        elif distance_between_rects(rect, self.head_biggest) < 40:
            return True
        return False

    def _one_rect_operations(self, rect):
        if close_to_each_other_central(self.last_rect, rect):
            if self.is_head(rect):
                rect = list(rect)
                #rect[0] += self.dx/4
                #rect[1] += self.dy/4
                self.last_rect = self.smooth_random_motions(rect)
            else:
                self.last_rect = self.smooth_random_motions(rect)
        else:
            if is_far_away(self.last_rect, rect):
                if close_to_each_other_central(rect, self.head_rect):
                    self.hand_info = -1
                elif rect[2]*rect[3]*4 <= self.last_rect[2]*self.last_rect[3]:
                    self.last_rect = combine_rects(self.last_rect, rect)
                    self.hand_info = -1
                else:
                    self.last_rect = rect
            elif rect[2]*rect[3]*4 <= self.last_rect[2]*self.last_rect[3]:
                self.last_rect = combine_rects(self.last_rect, rect)
                self.hand_info = -1
            elif not close_to_each_other_central(rect, self.head_rect):
                self.last_rect = rect

    def follow(self, img):
        draw_rects(img, [self.head_rect], 1, (0,255,0))
        if self.out:
            self.out_counter += 1
            if self.out_counter > self.out_limit:
                self.out = False
                self.out_counter = 0
                self.out_limit = 7
            return None, self.gesture

        self.shape_analysis()
        prediction = False
        if self.last_rect == None:
            if len(self.rects) == 0:
                return None, self.gesture
            if len(self.rects) == 1:
                if close_to_edge(self.rects[0]) and self.current_types[0] == "OPEN HAND":
                    self.last_rect = self.rects[0]
                    self.gesture = self.current_types[0]
            if len(self.rects) == 2:
                r1_close = close_to_edge(self.rects[0])
                r2_close = close_to_edge(self.rects[1])
                if r1_close and self.current_types[0] == "OPEN HAND":
                    self.last_rect = self.rects[0]
                    self.gesture = self.current_types[0]
                elif r2_close and self.current_types[1] == "OPEN HAND":
                    self.last_rect = self.rects[1]
                    self.gesture = self.current_types[1]
        else:
            if len(self.rects) == 0:
                prediction = True
            elif len(self.rects) == 1:
                self.gesture = self.current_types[0]
                self._one_rect_operations(self.rects[0])
            elif len(self.rects) == 2:
                r1_close = close_to_each_other_central(self.last_rect, self.rects[0])
                r2_close = close_to_each_other_central(self.last_rect, self.rects[1])
                if r1_close and r2_close:
                    r1_pred = close_to_each_other_central(self.predicted_rect, self.rects[0])
                    r2_pred = close_to_each_other_central(self.predicted_rect, self.rects[1])
                    r2_head = close_to_each_other_central(self.rects[1], self.head_rect)
                    r1_head = close_to_each_other_central(self.rects[0], self.head_rect)
                    if r1_pred and not r2_pred:
                        self.last_rect = self.rects[0]
                    elif r2_pred and not r1_pred:
                        self.last_rect = self.rects[1]
                    elif r1_pred and r2_pred:
                        if r2_head:
                            self.last_rect = self.rects[0]
                        elif r1_head:
                            self.last_rect = self.rects[1]
                elif r1_close:
                    self.gesture = self.current_types[0]
                    self._one_rect_operations(self.rects[0])
                elif r2_close:
                    self.gesture = self.current_types[1]
                    self._one_rect_operations(self.rects[1])
                else:
                    r1_close = False
                    r2_close = False
                    r3_close = False 
                    r4_close = False
                    if self.predicted_rect is not None:
                        r1_close = close_to_each_other_central(self.predicted_rect, self.rects[0])
                        r2_close = close_to_each_other_central(self.predicted_rect, self.rects[1])
                        small_len = len(self.rects_small)    
                        if small_len > 0:
                            r3_close = close_to_each_other_central(self.last_rect, self.rects_small[0])
                            if small_len > 1:
                                r4_close = close_to_each_other_central(self.last_rect, self.rects_small[1])
                    if r3_close:
                        self.last_rect = self.rects_small[0]
                    elif r4_close:
                        self.last_rect = self.rects_small[1]
                    elif r1_close and not r2_close:
                        self.last_rect = self.rects[0]
                        self.gesture = self.current_types[0]
                    elif r2_close and not r1_close:
                        self.last_rect = self.rects[1]
                        self.gesture = self.current_types[1]
                    elif r2_close and r1_close:
                        self.last_rect = self.rects[0]
                    elif self.current_types[0] not in ("FACE", "PALM", "UNKNOWN"):
                        self.last_rect = self.rects[0]
                        self.gesture = self.current_types[0]
                    elif self.current_types[1] not in ("FACE", "PALM", "UNKNOWN"):
                        self.last_rect = self.rects[1]
                        self.gesture = self.current_types[1]
                    else:
                        self.last_rect = self.predicted_rect
                    prediction = True
        self.analyze_rects(img)
        #special
        self.draw_info(img)
        #postprocessing
        if self.check_borders():
            self.clear()
            self.out = True
            return None, self.gesture
        self.prediction_limit(prediction)

        if self.last_rect:
            draw_rects(img, [self.last_rect], 2)
            return self.last_rect, self.gesture
        return None, self.gesture

    def draw_info(self, img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, self.gesture, (10, 25), font, 1, (255,255,255), 2)

    def draw_text(self, img, text, pos):
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, text, pos, font, 1, (255, 0, 255))