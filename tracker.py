"""
    Trackery, ktore wspolpracuja z background.py
    Najnowszy tracker do swiatla dziennego:
        TrackerNext
        TODO: 
        * aproksymacja kilku bounding boxow
            - wez najwiekszy, jesli jest w poblizu kilka innych to polacz
            - jesli kolo najwiekszego nie ma innych moze mozna przeskoczyc na inny
        * predykcja ruchu i wyznaczanie boxow z prawdopodobienstwem     

    StateTracker TODO:
    - rozwiazac sporadycznie pojawiajace sie problemy z przechodzeniem reki nad twarza/wyrzucaniem sie aplikacji
    - napisac od nowa BodyPartsModel
    - convectivyDefects
    - kalibracja
"""


import cv2
import numpy as np
from copy import copy
from collections import deque

from main_utils import draw_circles, \
    draw_rects, \
    close_to_edge, \
    close_to_each_other, \
    is_far_away, \
    is_big_enough, \
    one_inside_another, \
    CFG_HEIGHT, CFG_WIDTH, \
    draw_circles, \
    average_rect, \
    average_queue, \
    is_near_rect, \
    correct_rect, \
    distance_between_rects
from shape_discovery import ShapeDiscovery


BODY_PARTS = ["UNKNOWN", "HAND", "FACE", "FACE_AND_HAND"]

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
            
            
#############################################################################################


class TrackerNext:

    def __init__(self):
        self.x = -1
        self.y = -1
        self.vx = 0
        self.vy = 0
        self.predicted_x = -1
        self.predicted_y = -1
        self.sizex_up = 60
        self.sizex_down = 70
        self.sizey_up = 180
        self.sizey_down = 30
        self.the_box = [-1,-1,-1,-1]
        self.last_positions = []
        self.enter_counter = 0
        self.jump_counter = 0
            
    def draw_bounding_rects(self, img, contours):
        cv2.line(img, (30, 10),(30, 70),(255,0,0))
        cv2.line(img, (30, 10),(90, 10),(255,0,0))
        cnts = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 500]
        for cnt in cnts:
            x,y,w,h = cnt
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 1)
            
    def draw_rect(self, img, box):
        x,y,w,h = box
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 3)
        
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
        
    def process_bounding_rects(self, contours):
        sizex_up = self.sizex_up
        sizex_down = self.sizex_down
        sizey_up = self.sizey_up
        sizey_down = self.sizey_down
        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 500]
        found = [0,0,0,0]
        #when there is more of small bounding boxes
        for r in rects:
            if r[2]*r[3] > found[2]*found[3]:
                found = list(r)
        if found[2] == 0:
            found = False
        elif found[2]*found[3] < 4800:
            found[0] -= 20
            found[1] -= 20
            found[2] += 60
            found[3] += 80
        return found
        
    def track_rect(self, best_box):
        enter_x = 80
        enter_y = 80
        exit_x = 40
        exit_y = 40
        jump_constr = 200
        jump_thr_x = 120
        jump_thr_y = 120
        box_size_inertia_w = 15
        box_size_inertia_h = 20
        if not best_box:
            return self.the_box
        cx = best_box[0] + best_box[2]/2
        cy = best_box[1] + best_box[3]/2
        if self.the_box[3] == -1:    
            if cx < enter_x or cx > 640 - enter_x or cy > 480 - enter_y:
                self.the_box = best_box
        else:
            if (cx < exit_x and self.vx < -exit_x) or (cx > 640 - exit_x and self.vy > exit_y ) or (cy > 480 - exit_y and self.vx > exit_x):
                self.the_box = [-1,-1,-1,-1]
            else:
                diff_x = self.the_box[0] - best_box[0]
                diff_y = self.the_box[1] - best_box[1]
                diff_w = self.the_box[2] - best_box[2]
                diff_h = self.the_box[3] - best_box[3]
                if abs(diff_x) > jump_constr or abs(diff_y) > jump_constr:
                    return self.the_box
                if abs(diff_x) > jump_thr_x:
                    best_box[0] = best_box[0] + diff_x/2
                if abs(diff_y) > jump_thr_y:
                    best_box[1] = best_box[1] + diff_y/2
                if abs(diff_w) > box_size_inertia_w:
                    if diff_w > 0:
                        best_box[2] = best_box[2] + diff_w - box_size_inertia_w
                    else:
                        best_box[2] = best_box[2] + diff_w + box_size_inertia_w
                if abs(diff_h) > box_size_inertia_h:
                    if diff_h > 0:
                        best_box[3] = best_box[3] + diff_h - box_size_inertia_w
                    else:
                        best_box[3] = best_box[3] + diff_h + box_size_inertia_w
                #zapamietanie ostatnich pozycji + predykcja ruchu
                self.the_box = best_box
        return self.the_box
    
    def update(self, img): 
        cp = img.copy()
        contours, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        self.draw_bounding_rects(img, contours)
        best_box = self.process_bounding_rects(contours)
        the_rect = self.track_rect(best_box)
        self.draw_rect(img, the_rect)
   

##########################################################################################################################################     


class StateTracker(object):
    
    def __init__(self):
        self.clear()
        self.out_limit = 20
        self.rsave = RectSaver()
        self.dsc = ShapeDiscovery()
        self.head_rect = [200, 200, 0, 0]

    def clear(self):
        self.last_rect = None
        self.before_last_rect = None
        self.really_old_rect = None

        self.predicted_rect = None
        self.prediction_counter = 0
        self.before_prediction_rect = None

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

    def choose_contour(self, contours):
        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 2400]
        def area(rect):
            return 1/float(rect[2]*rect[3])
        s_rects = sorted(rects, key=area)
        return rects[:2]
        
    def update(self, img):
        """things that have to be done before selection of new rect"""
        cp = img.copy()
        contours, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.rects = self.choose_contour(contours)
        self.predicted_rect = self.prediction_calculations()
        self.average_calculations()
        self.really_old_rect = self.before_last_rect
        self.before_last_rect = self.last_rect
    
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
        if self.is_not_real_counter > 6 and (x < 30 or x + w > CFG_WIDTH - 30 or y < 30 or y + h > CFG_HEIGHT - 30):
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
        elif one_inside_another(candidate, self.really_old_rect, 2):
            return self.really_old_rect
            #return self.last_rect
        else:
            return candidate


    def prediction_limit(self, prediction):
        if prediction:
            self.prediction_counter += 1
        else:
            self.prediction_counter = 0
            self.before_prediction_rect = self.last_rect
        if self.prediction_counter >= 2 and self.last_rect is not None:
            self.last_rect = self.before_prediction_rect


    def fit_rect_size(self):
        if self.last_rect is None:
            return
        self.last_rect = list(self.last_rect)
        #jezeli dwa boxy na siebie nachodza to natnij box reki
        #x+w > xg ale x < xg
        #x < xg + wg ale x+w > xg + wg
        x,y,w,h = self.last_rect
        hx,hy,hw,hh = self.head_rect
        primary_condition_x = x + w > hx and x < hx
        primary_condition_y = y + h > hy and y < hy
        if primary_condition_x and w > 300:
            diff = x + w - hx
            self.last_rect[2] = w - diff/2
            self.last_rect[0] = x + diff/2
        if primary_condition_y and primary_condition_x and h > 300:
            diff = y + h - hy
            self.last_rect[1] = y + diff/3
            self.last_rect[3] = h - 2*diff/3
        if self.last_rect[3] > CFG_HEIGHT/1.5:
            self.last_rect[3] = int(self.last_rect[3]/1.5)

    def analyze_rects(self, img):
        _type = "UNKNOWN"
        if self.last_rect is not None:
            dsc_rect = correct_rect(self.last_rect)
            val = self.dsc.discover(img, dsc_rect)
            _type = BODY_PARTS[val]
        elif len(self.rects) == 1:
            dsc_rect = correct_rect(self.rects[0])
            val = self.dsc.discover(img, dsc_rect)
            _type = BODY_PARTS[val]
        print _type, #####################

        is_real = self.dsc.is_real()
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
            if self.last_rect is None and _type == "FACE":
                self.head_rect = average_rect(self.head_rect, self.rects[0])
        elif len(self.rects) == 2:
            if self.rects[0] == self.last_rect and not is_near_rect(self.last_rect, self.rects[1]):
                self.head_rect = average_rect(self.head_rect, self.rects[1])
            elif self.rects[1] == self.last_rect and not is_near_rect(self.last_rect, self.rects[0]):
                self.head_rect = average_rect(self.head_rect, self.rects[0])
            if is_real:
                self.is_not_real_counter = 0
            else:
                self.is_not_real_counter += 1


    def follow(self, img):
        draw_rects(img, [self.head_rect], 2, (0,255,0))
        if self.out:
            self.out_counter += 1
            if self.out_counter > self.out_limit:
                self.out = False
                self.out_counter = 0
                self.out_limit = 10
            return

        prediction = False
        if self.last_rect == None:
            if len(self.rects) == 0:
                return
            if len(self.rects) == 1:
                if close_to_edge(self.rects[0]):
                    self.last_rect = self.rects[0] 
            if len(self.rects) == 2:
                r1_close = close_to_edge(self.rects[0])
                r2_close = close_to_edge(self.rects[1])
                if r1_close:
                    self.last_rect = self.rects[0]
                elif r2_close:
                    self.last_rect = self.rects[1]
        else:
            if len(self.rects) == 0:
                self.last_rect = self.predicted_rect
                prediction = True
            elif len(self.rects) == 1:
                if close_to_each_other(self.last_rect, self.rects[0]):
                    self.last_rect = self.smooth_random_motions(self.rects[0])
                else:
                    if is_far_away(self.last_rect, self.rects[0]):
                        prediction = True
                    elif is_big_enough(self.rects[0]):
                        self.last_rect = self.rects[0]
            elif len(self.rects) == 2:
                #self.rsave.save_hand(img, self.rects[0], self.rects[1])
                r1_close = close_to_each_other(self.last_rect, self.rects[0])
                r2_close = close_to_each_other(self.last_rect, self.rects[1])
                if r1_close and r2_close:
                    if one_inside_another(self.rects[0], self.head_rect, 2):
                        self.last_rect = self.rects[1]
                    elif one_inside_another(self.rects[1], self.head_rect, 2):
                        self.last_rect = self.rects[0]
                elif r1_close:
                    print "r1 blisko"
                    self.last_rect = self.smooth_random_motions(self.rects[0])
                elif r2_close:
                    print "r2 blisko"
                    self.last_rect = self.smooth_random_motions(self.rects[1])
                else:
                    print "r2 i r1 daleko"
                    self.last_rect = self.predicted_rect 
                    prediction = True
        self.analyze_rects(img)

        #postprocessing
        if self.check_borders():
            self.clear()
            self.out = True
            return
        self.prediction_limit(prediction)
        self.fit_rect_size()
        if self.last_rect:
            draw_rects(img, [self.last_rect], 2)

            