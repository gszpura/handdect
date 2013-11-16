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
    
def is_far_away(rect1, rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2
    far_away_x = min(int(1.2*w1), CFG_FAR_AWAY_X)
    far_away_y = min(2*h1, CFG_FAR_AWAY_Y)
    if abs(x2 - x1) >= far_away_x or abs(y2 - y1) >= far_away_y:
        return True
    return False
    
def is_big_enough(rect1):
    x1,y1,w1,h1 = rect1
    if w1*h1 >= 9600:
        return True
    return False
    
    
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
    sum = int(left) + int(right) + int(up) + int(down)
    if rigid and sum == 4:
        return True
    if sum >= 3:
        current_area = current[2]*current[3]
        previous_area = previous[2]*previous[3]
        if previous_area > ratio*current_area:
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
"""
KOSZ NA ODPADY:

def check_borders(self):
        if self.predicted_rect is None or self.last_rect is None:
            return False
        if self.predicted_rect[0] < 0 or self.predicted_rect[0] > CFG_WIDTH - 25 or self.predicted_rect[1] > CFG_HEIGHT - 25:
            self.predicted_rect = None
            self.last_rect = None
            return True
        return False




        c2, c1  - kontury
        h_c2, h_c1 - obrazy czarno - biale
        #e2 = cv2.fitEllipse(c2)
        #e1 = cv2.fitEllipse(c1)
        #cv2.ellipse(h_c2, e2, (255,0,0))
        #cv2.ellipse(h_c1, e1, (255,0,0))


    def calc_arc_shape(self, edge):
        ln = len(edge)
        cnt = 0
        outcome = []
        for i in range(0,ln-1):
            fst = edge[i]
            sec = edge[i+1]
            if fst - sec > 0:
                cnt += 1
            elif fst - sec < 0:
                cnt -= 1
            if cnt >= 3:
                outcome.append(1)
                cnt = 0
            if 1 in outcome and cnt < -4:
                outcome.append(-1)
                cnt = 0
        outcome = set(outcome)
        if len(outcome) == 2:
            return True
        return False



DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

cdef class LeafNum:

    cdef np.ndarray slice
    cdef np.ndarray footprint

    cdef int slice_len
    cdef int footprint_len

    cdef int first
    cdef int last

    cdef int length

    def __init__(self, np.ndarray[DTYPE_t, ndim=1] sl):
        self.slice = sl
        self.first = BLACK
        self.last = BLACK
        self.length = 0
        self.slice_len = len(sl)
        self.footprint = np.zeros([20], dtype=DTYPE)
        self.footprint_len = 0
        self.stats()

    cdef void stats(self):
        cdef int iter_range = self.slice_len - 1 
        cdef int max = 255
        #self.first = max in self.slice[0:5] and WHITE or BLACK 
        #self.last = max in self.slice[-5:-1] and WHITE or BLACK
        cdef int index = 0
        cdef DTYPE_t cnt, current
        cnt = 1
        current = self.slice[1] & 0x01
        cdef unsigned int i
        for i in xrange(1, iter_range):
            if self.slice[i] & 0x01 != current:
            #   current ^= 0x01
            #   self.footprint[index] = cnt
            #   cnt = 1
            #   index += 1
            #else:
            #   cnt = cnt + 1
                pass
        #self.footprint_len = self.footprint.shape[0]


cpdef fun(int az):
    cdef int z = az
    cdef int b
    cdef unsigned int i
    cdef object a = [1,2,3]
    for i in xrange(0,10000000):
        b = z*z
        b = 4/3
        b = b + 1
        len(a)


cpdef fun2(int az):
    cdef int z = az
    cdef int b
    cdef unsigned int i
    cdef list a = [1,2,3]
    for i in xrange(0,10000000):
        b = z*z
        b = 4/3
        b = b + 1
        len(a)

cpdef fun3(int az):
    cdef int z = az
    cdef int b
    cdef unsigned int i
    cdef np.ndarray a = np.array([1, 2, 3], dtype=np.int)
    for i in xrange(0,10000000):
        b = z*z
        b = 4/3
        b = b + 1
        a.shape[0]

"""