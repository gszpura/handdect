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
    - box czasem zostaje przy bardzo szybkim ruchu? [poprawic czy zostawic]
    - bezwladnosc rozmiarow boxa
    - zle wychodzenie reki poza dolna krawedz - przyczyny:
        * prediction counter
        * x,y uzyty z lewego gornego rogu
        * mozna uzyc bezwladnosci rozmiarow boxa i szacowanej pozycji, box sie ciagle skraca przy dochodzeniu do kraweedzi
    - pick_better_box ma korzystac z predyktora ruchu
    - pick_better_box powinien korzystac z momentow blobu oraz z mechanizmu rozpoznawania twarzy
    - zaimplementowac mechanizm rozpoznawania twarzy
    - mozna zaimplementowac mechanizm wypelniania z sasiadami
        
"""


import cv2
from copy import copy
from main_utils import draw_circles, \
    draw_rects, \
    close_to_edge, \
    close_to_each_other, \
    is_far_away, \
    is_big_enough, \
    one_inside_another, \
    CFG_HEIGHT, CFG_WIDTH
from collections import deque

class Tracker:

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
        self.the_group = [-1,-1,-1,-1]
        self.enter_counter = 0
        self.jump_counter = 0
            
    def draw_bounding_boxes(self, img, contours):
        cnts = [cv2.boundingRect(cnt) for cnt in contours if 500 < cv2.contourArea(cnt) < 20000]
        for cnt in cnts:
            x,y,w,h = cnt
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0), 1)
            
    def draw_groups(self, img, groups):
        for g in groups:
            x,x2,y,y2 = g
            print g
            cv2.rectangle(img, (x,y),(x2,y2), (255,0,0), 2)
    
    def draw_single_group(self, img, group):
        x,x2,y,y2 = group
        cv2.rectangle(img, (x,y),(x2,y2), (255,0,0), 2)
            
        
    def process_bounding_boxes(self, contours):
        sizex_up = self.sizex_up
        sizex_down = self.sizex_down
        sizey_up = self.sizey_up
        sizey_down = self.sizey_down
        rects = [cv2.boundingRect(cnt) for cnt in contours if 500 < cv2.contourArea(cnt) < 20000]
        rects2 = sorted(rects)

        groups = []
        change = False
        change_cnt = 0
        for rect in rects2:
            if len(groups) == 0:
                groups.append([rect[0]-sizex_down, rect[0]+sizex_up, rect[1]-sizey_down, rect[1]+sizey_up])
                continue
            change_cnt = 0
            for g in groups:
                if g[0] < rect[0] < g[1] and g[2] < rect[1] < g[3]:
                    change = True
                    break
                change_cnt += 1
            if change:
                x2 = groups[change_cnt][1]
                y1 = groups[change_cnt][2]
                y2 = groups[change_cnt][3]
                if rect[0] > x2 - sizex_up:  
                    groups[change_cnt][1] = rect[0] + sizex_up
                if rect[1] < y1 + sizey_down:
                    groups[change_cnt][2] = rect[1] - sizey_down
                elif rect[1] > y2 - sizey_up:
                    groups[change_cnt][3] = rect[1] + sizey_up
            elif len(groups) < 3:
                groups.append([rect[0]-sizex_down, rect[0]+sizex_up, rect[1]-sizey_down, rect[1]+sizey_up])
            change = False
        return groups
        
    def best_group(self, groups):
        the_g = self.the_group
        diffx = 60
        diffy = 60
        jump_limit_x = 130
        jump_limit_y_up = 90
        jump_limit_y_down = 150
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
                score = abs(the_g[0] - group[0]) + abs(the_g[2] - group[2])
                if score < best_score:
                    best_match = group
                    best_score = score
                    y_jump = group[2] - the_g[2]
        #prosta implementacja wstrzymywania sie z dlugim skokiem w dol obrazu
        if y_jump and y_jump > jump_limit_y_up/2:
            if self.jump_counter > 1:
                self.jump_counter = 0
                return best_match
            else:
                self.jump_counter += 1
                return False
        return best_match
    
    def update(self, img):
        w = 40; h = 40    
        cp = img.copy()
        contours, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        #cnts = [(cnt, cv2.moments(cnt)) for cnt in contours if 500 < cv2.contourArea(cnt) < 6000]
        #positions = [(int(cnt[1]['m10']/cnt[1]['m00']), int(cnt[1]['m01']/cnt[1]['m00'])) for cnt in cnts]
        
        #draw_circles(img, positions)
        self.draw_bounding_boxes(img, contours)
        groups = self.process_bounding_boxes(contours)
        
        #self.draw_groups(img, groups)
        print groups
        best_group = self.best_group(groups)
        if best_group:
            self.the_group = best_group
        if self.the_group:
            self.draw_single_group(img, self.the_group)
            
            
            
            
###########################################################################################################            
            
            
            
            
            
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
            print g
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
            #print found
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
        print the_rect
        self.draw_rect(img, the_rect)
        
        
class StateTracker(object):
    
    def __init__(self):
        self.last_rect = None
        self.before_last_rect = None
        self.really_old_rect = None

        self.predicted_rect = None
        self.prediction_counter = 0
        self.before_prediction_rect = None
        
        self.enter_last = []
        self.enter_before = []
        self.enter_state = 0
        
        self.last_v = (0,0)
        self.rects = []
        
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

    def choose_contour(self, contours):
        rects = [cv2.boundingRect(cnt) for cnt in contours if cv2.contourArea(cnt) > 2400]
        def area(rect):
            return 1/float(rect[2]*rect[3])
        s_rects = sorted(rects, key=area)
        return rects[:2]
        
    def update(self, img):
        cp = img.copy()
        contours, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        self.rects = self.choose_contour(contours)
        draw_rects(img, self.rects, 2)
        
    def pick_better_rect(self):
        if len(self.rects) == 1:
            return self.rects[0]
        points1 = 0
        points2 = 0
        #area
        a1 = self.rects[0][2]*self.rects[0][3]
        a2 = self.rects[1][2]*self.rects[1][3]
        return self.last_rect
    
    def internal_one_inside_another(self):
        left = self.last_rect[0] >= self.before_last_rect[0]-5
        right = self.last_rect[0] + self.last_rect[2] <= self.before_last_rect[0] + self.before_last_rect[2]+5
        up = self.last_rect[1] >= self.before_last_rect[1]-5
        down = self.last_rect[1] + self.last_rect[3] <= self.before_last_rect[1] + self.before_last_rect[3]+5
        if left and  right and up and down:
            return True
        sum = int(left) + int(right) + int(up) + int(down)
        if sum >= 3:
            area_last = self.last_rect[2]*self.last_rect[3]
            area_before_last = self.before_last_rect[2]*self.before_last_rect[3]
            if area_before_last > 3*area_last:
                return True
        return False
        
    def outside_scope(self):
        if self.last_rect[0] < 0 or self.last_rect[0] > CFG_WIDTH or self.last_rect[1] < 0 or self.last_rect[1] > CFG_HEIGHT:
            return True
        return False
        
    def check_borders(self):
        if self.predicted_rect is None or self.last_rect is None:
            return False
        if self.predicted_rect[0] < 0 or self.predicted_rect[0] > CFG_WIDTH - 25 or self.predicted_rect[1] > CFG_HEIGHT - 25:
            self.predicted_rect = None
            self.last_rect = None
            return True
        return False
        
    def speed_prediction(self):
        if self.before_last_rect == None:
            self.before_last_rect = self.last_rect
            return self.last_rect
        else:
            if self.internal_one_inside_another():
                return self.before_last_rect
            if self.outside_scope():
                return None
            dx = self.last_rect[0] - self.before_last_rect[0]
            dy = self.last_rect[1] - self.before_last_rect[1]
            w = (self.last_rect[2] + self.before_last_rect[2])/2
            h = (self.last_rect[3] + self.before_last_rect[3])/2
            x = self.last_rect[0]+dx
            y = self.last_rect[1]+dy
            
            self.really_old_rect = self.before_last_rect
            self.before_last_rect = self.last_rect
            self.predicted_rect = [x, y, w, h]
            return self.predicted_rect


    def smooth_random_motions(self, candidate):
        if one_inside_another(candidate, self.last_rect):
            return self.last_rect
        elif one_inside_another(candidate, self.really_old_rect, 3.2):
            return self.last_rect
        else:
            return candidate


    def prediction_limit(self, prediction):
        if prediction:
            self.prediction_counter += 1
        else:
            self.prediction_counter = 0
            self.before_prediction_rect = self.last_rect
        if self.prediction_counter >= 2:
            self.last_rect = self.before_prediction_rect
            
        
    def fit_rect_size(self):
        if self.last_rect is None:
            return
        self.last_rect = list(self.last_rect)
        if self.last_rect[3] > CFG_HEIGHT/1.5:
            self.last_rect[3] = int(self.last_rect[3]/1.5)
        try:
            self.last_rect[3] = 0.6*self.before_last_rect[3] + 0.4*self.last_rect
            self.last_rect[2] = 0.6*self.before_last_rect[2] + 0.4*self.last_rect
        except:
            pass
        

    def follow(self, img):
        prediction = False
        if self.last_rect == None:
            if len(self.rects) == 0:
                return
            if len(self.rects) == 1:
                if close_to_edge(self.rects[0]):
                    self.last_rect = self.rects[0]
                    #check if it's face?
            if len(self.rects) == 2:
                r1_close = close_to_edge(self.rects[0])
                r2_close = close_to_edge(self.rects[1])
                if r1_close and r2_close:
                    self.last_rect = self.pick_better_rect()
                elif r1_close:
                    self.last_rect = self.rects[0]
                elif r2_close:
                    self.last_rect = self.rects[1]
        else:
            self.predicted_rect = self.speed_prediction()
            if len(self.rects) == 0:
                self.last_rect = self.predicted_rect
                prediction = True
            elif len(self.rects) == 1:
                if close_to_each_other(self.last_rect, self.rects[0]):
                    self.last_rect = self.smooth_random_motions(self.rects[0])
                else:
                    if is_far_away(self.last_rect, self.rects[0]):
                        self.last_rect = self.predicted_rect
                        prediction = True
                    elif is_big_enough(self.rects[0]):
                        #check if face? or if its big enough
                        self.last_rect = self.rects[0]
            elif len(self.rects) == 2:
                r1_close = close_to_each_other(self.last_rect, self.rects[0])
                r2_close = close_to_each_other(self.last_rect, self.rects[1])
                if r1_close and r2_close:
                    self.last_rect = self.pick_better_rect()
                elif r1_close:
                    self.last_rect = self.smooth_random_motions(self.rects[0])
                elif r2_close:
                    self.last_rect = self.smooth_random_motions(self.rects[1])
                else:
                    self.last_rect = self.predicted_rect 
                    prediction = True
                    #self.last_rect = self.pick_better_rect()
                  
        if self.check_borders():
            return
        self.prediction_limit(prediction)
        self.fit_rect_size()
        if self.last_rect:
            draw_rects(img, [self.last_rect], 2)
            