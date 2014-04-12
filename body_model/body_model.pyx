"""
BodyModel.

1. Implements lattice algorithm
2. It's a hand and face model: can distinguish between them.
"""

import cython
cimport cython
import cv2
import numpy as np
cimport numpy as np
from cpython cimport bool

cdef short int WHITE = 1
cdef short int BLACK = 0


cdef int UNKNOWN = 0
cdef int OPEN_HAND = 1
cdef int ONE_FINGER = 2
cdef int THUMB = 3
cdef int PALM = 4
cdef int FACE = 5
cdef int TWO_FINGERS = 6

cdef int HORIZONTAL_JUMP = 10
cdef int VERTICAL_JUMP = 8



cdef int sign(int elem):
    if elem > 0:
        return 1
    elif elem < 0:
        return -1
    return 0


cdef class Leaf:
    """
    Represents a single line in the image.
    For example, we have image of size 100x100 which
    depicts hand.
    Leaf is a part of that image. It's a line 1x100
    (one pixel high and 100 pixels wide).


    Leaf has it's footprint.
    Footprint of a leaf is a trace of white and black
    regions inside it. For example we have a Leaf:
    [255, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255]
    where 255 stands for white and 0 stands for black.
    Footprint of this leave will be: [6, 3].
    It means that first region in the Leaf is 6 pixels wide
    and second is 3 pixels wide. Third region is
    len(Leaf) - 9 pixels wide.
    """

    cdef list slice
    cdef public list footprint

    cdef public int slice_len
    cdef public int footprint_len

    cdef public short int first
    cdef public short int last

    cdef public short int all_white
    cdef bool vertical

    def __init__(Leaf self, list sl, bool vertical=False):
        self.slice = sl
        self.first = BLACK
        self.last = BLACK
        self.slice_len = len(sl)
        self.footprint = []
        self.footprint_len = 0
        self.vertical = vertical
        self.stats()

    @cython.boundscheck(False)
    cdef void stats(self):
        """
        Finds footprint of a leaf.
        Footprint [100,20] and first == BLACK
        means that we have 3 layers
        first is BLACK and 100 pixels thick
        second is WHITE and 20 pixels thick
        third is BLACK and self.slice_len - 120 pixels thick
        """
        cdef unsigned int iter_range = self.slice_len - 1
        cdef int max = 255
        self.first = max in self.slice[0:5] and WHITE or BLACK 
        self.last = max in self.slice[-5:-1] and WHITE or BLACK
        cdef int cnt, current
        cnt = 1
        current = self.slice[1] & 0x01
        cdef unsigned int i
        for i in xrange(1, iter_range):
            if self.slice[i] & 0x01 != current:
                current ^= 0x01
                self.footprint.append(cnt)
                cnt = 1
            else:
                cnt = cnt + 1
        self.footprint_len = len(self.footprint)

        self.all_white = 0
        if self.vertical:
            if (self.footprint_len == 0 and self.first == WHITE) or \
               (self.footprint_len == 1 and self.slice.count(255) > 0.95*self.slice_len): 
                self.all_white = 1


cdef class LeafStats:
    
    cdef list leafs

    cdef bool open_hand_shape
    cdef bool two_fingers_shape
    cdef bool face_shape

    cdef double edge_dev
    cdef double thickness_dev
    cdef double thickness_avg

    cdef bool up_down_white
    cdef bool fluctuations
    cdef list distribution
    cdef list edge

    def __init__(LeafStats self, list leafs):
        self.leafs = leafs
        self.stats()    

    cdef list init_distribution(LeafStats self):
        cdef list distribution = [0]*30
        return distribution

    cdef bool calc_open_hand_shape(LeafStats self, list distribution):
        """
        Checks if the image matches an OPEN HAND shape.
        Vast majority of the leafs must have more than 4 regions.
        """
        cdef int all_leafs = sum(distribution)
        cdef int part_value = int(0.8*all_leafs)
        cdef bool fingers = False
        
        fingers = sum(distribution[4:15]) > part_value
        return fingers

    cdef bool calc_two_fingers_shape(LeafStats self, list distribution):
        """
        Checks if the image matches a TWO FINGERS shape.
        Majority of the leafs must have exactly two white regions
        and it means 5 regions in total (add 3 black regions).
        """
        cdef int all_leafs = sum(distribution)
        cdef int threshold_for_two_fingers = int(0.4*all_leafs)
        cdef bool two_fingers = False

        two_fingers = sum(distribution[5:6]) > threshold_for_two_fingers
        return two_fingers

    cdef bool calc_face_shape(LeafStats self, list distribution):
        cdef int all_leafs = sum(distribution)
        cdef int v1 = int(0.7*all_leafs)
        if sum(distribution[1:4]) >= v1:
            return True
        return False

    cdef double calc_edge_dev(LeafStats self, list edge):
        """
            Calculates first edge deviation in percentages.
            Observation reveals that if edge_dev is more
            than 20% it is unlikely that image is a face.
        """
        if edge:
            return np.std(edge)/self.leafs[0].slice_len
        return 0.0

    cdef double calc_thickness_dev(LeafStats self, list thickness):
        cdef int _len = len(thickness)
        cdef list s_thick
        if _len >= 4:
            s_thick = sorted(thickness)
            s_thick = s_thick[3:]
            return np.std(s_thick)/self.leafs[0].slice_len
        return 0.0

    cdef double calc_thickness_avg(LeafStats self, list thickness):
        cdef int _len = len(thickness)
        if _len >= 3:
            return np.average(thickness)
        return 0.0

    cdef bool calc_fluctuations(LeafStats self, list edge):
        cdef unsigned int i
        cdef unsigned int length = len(edge)
        cdef int counter = 0
        cdef int grad = sign(edge[1] - edge[0])
        cdef int tmp = 0

        for i in xrange(0, length-1):
            tmp_sign = sign(edge[i+1] - edge[i])
            if  tmp_sign != grad and tmp_sign != 0:
                counter += 1
                grad = tmp_sign
        if counter > 3:
            return True
        return False


    cdef void stats(self):
        cdef list distribution = self.init_distribution()
        cdef list edge = []
        cdef list thickness = []  
        
        cdef list tmp_leafs = self.leafs
        cdef unsigned int length = len(tmp_leafs)
        cdef unsigned int i, j

        #variable for vertical leafs stats
        cdef int up_down_count = 0

        #internal loop variables
        cdef Leaf leaf
        cdef short int fst, lst
        cdef list footprint
        cdef int footprint_len
        for i in xrange(0, length):
            leaf = tmp_leafs[i]
            fst = leaf.first
            lst = leaf.last
            footprint = leaf.footprint
            footprint_len = leaf.footprint_len
            distribution[footprint_len+1] += 1
            if footprint_len == 0 and fst == WHITE:
                edge.append(0)
            elif footprint_len > 0:
                edge.append(footprint[0])
            if footprint_len == 1 and fst == WHITE:
                thickness.append(footprint[0])
            elif footprint_len == 1 and fst == BLACK:
                thickness.append(leaf.slice_len - footprint[0])
            if footprint_len == 2 and fst == BLACK:
                thickness.append(footprint[1])
            #vertical
            if leaf.all_white:
                up_down_count += 1
        self.distribution = distribution
        self.edge = edge

        self.thickness_dev = self.calc_thickness_dev(thickness)
        self.edge_dev = self.calc_edge_dev(edge)
        self.thickness_avg = self.calc_thickness_avg(thickness)

        self.open_hand_shape = self.calc_open_hand_shape(distribution)
        self.two_fingers_shape = self.calc_two_fingers_shape(distribution)
        self.face_shape = self.calc_face_shape(distribution)
            
        self.up_down_white = False
        if up_down_count >= 2:
            self.up_down_white = True
        self.fluctuations = self.calc_fluctuations(edge)


cdef class BodyPartsModel(object):

    cdef np.ndarray img
    cdef np.ndarray debug_img
    cdef int outcome

    cdef int w
    cdef int h

    def __init__(self, img):
        self.img = img
        if img is not None:
            self.w = img.shape[1]
            self.h = img.shape[0]
        self.outcome = UNKNOWN
        self.process()

    cpdef int get_value(self):
        return self.outcome

    cdef bool has_not_enough_information(self):
        cdef np.ndarray[np.uint8_t, ndim=2] img = self.img
        cdef int nonZero, size = 0
        if self.img is None:
            return True
        size = img.shape[0]*img.shape[1]
        if size < 4000:
            return True
        if size < 10000:
            nonZero = cv2.countNonZero(img)
            if nonZero > int(0.5*size):
                return False
            return True
        return False


    cdef list get_upper_leafs(self):
        cdef np.ndarray[np.uint8_t, ndim=2] img = self.img
        cdef int amount = img.shape[0]/HORIZONTAL_JUMP
        cdef int fst_jumps = amount/2
        cdef int c = 20
        cdef list leafs = []

        cdef unsigned int i, j
        cdef np.ndarray[np.uint8_t, ndim=1] sl
        cdef int JUMP = 2*HORIZONTAL_JUMP/3
        for i in xrange(0, fst_jumps):
            sl = img[c:c+1][0].copy()
            #self.debug_img[c:c+1][0] = 0
            c += JUMP
            leafs.append(Leaf(sl.tolist()))
        return leafs

    cdef list get_lower_leafs(self):
        cdef np.ndarray[np.uint8_t, ndim=2] img = self.img
        cdef int amount = img.shape[0]/HORIZONTAL_JUMP
        cdef int sec_jumps = amount/2
        cdef int c = img.shape[0]/2
        cdef list leafs = []

        cdef unsigned int j
        cdef np.ndarray[np.uint8_t, ndim=1] sl
        for j in xrange(0, sec_jumps):
            sl = img[c:c+1][0].copy()
            #img[c:c+1][0] = 0
            c += HORIZONTAL_JUMP
            leafs.append(Leaf(sl.tolist()))
        return leafs

    cdef list get_vertical_leafs(self):
        cdef np.ndarray[np.uint8_t, ndim=2] img = self.img
        cdef list leafs = []
        cdef int how_many = int(0.8*self.w/(VERTICAL_JUMP))
        cdef int amount = min(14, how_many)
        cdef int c = self.w/5

        cdef unsigned int i
        cdef np.ndarray[np.uint8_t, ndim=1] sl
        for i in range(0, amount):
            shp = img[:,c:c+1].shape
            #img[:c,c+1] = 0
            sl = img[:,c:c+1].reshape(shp[0])
            c += VERTICAL_JUMP
            leafs.append(Leaf(sl.tolist(), True))
        return leafs

    cdef int process(self):
        if self.has_not_enough_information():
            return self.outcome
        cdef list uleafs, lleafs, vleafs
        cdef LeafStats stats_u, stats_l, stats_v

        #self.debug_img = self.img.copy()
        vleafs = self.get_vertical_leafs()
        lleafs = self.get_lower_leafs()
        uleafs = self.get_upper_leafs()
        stats_l = LeafStats(lleafs)
        stats_u =  LeafStats(uleafs)
        stats_v = LeafStats(vleafs)

        if stats_l.open_hand_shape or stats_u.open_hand_shape:
            self.outcome = OPEN_HAND
            if stats_u.two_fingers_shape:
                #cv2.imshow('P', self.debug_img)
                self.outcome = TWO_FINGERS
        elif stats_l.thickness_avg*0.4 > stats_u.thickness_avg and stats_u.thickness_avg > 0.0:
            self.outcome = ONE_FINGER
        elif stats_l.face_shape and stats_u.face_shape:
            if 0.40 > stats_v.edge_dev > 0.16 and stats_v.thickness_dev > 0.10:
                self.outcome = THUMB
            elif stats_v.fluctuations:
                self.outcome = PALM
            else:
                self.outcome = FACE
        elif stats_v.up_down_white:
            self.outcome = FACE
        else:
            self.outcome = UNKNOWN

