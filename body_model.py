"""
	TODO:
	

"""

import cv2
import numpy as np
import time
from collections import deque

WHITE = 1
BLACK = 0

HAND = 1
HEAD = 2
HEAD_AND_HAND = 3
UNKNOWN = 0
BODY_PARTS = ["UNKNOWN", "HAND", "HEAD", "HEAD_AND_HAND"]

HORIZONTAL_JUMP = 10
VERTICAL_JUMP = 20


def check_jumps(lst):
    length = len(lst)
    if length < 4:
        return 0
    _sum = 0
    jumps = 0
    avg = 999
    for i in range(1, length):
        _curr = lst[i] - lst[i-1]
        if _curr > max(2*avg, 2):
            jumps += 1
        if _curr < -50:
        	jumps -= 1
        _sum = _sum + _curr
        avg = _sum/i
    return max(jumps, 0)


class Leaf(object):

	def __init__(self, sl):
		self.slice = sl
		self.len = 0
		self.first = BLACK
		self.last = BLACK
		self.footprint = []
		self.stats()

	def stats(self):
		sl = self.slice  
		ln = len(sl)
		self.first = 255 in sl[0:5] and WHITE or BLACK 
		self.last = 255 in sl[-5:-1] and WHITE or BLACK
		cnt = 1
		current = sl[1] & 0x01
		for i in xrange(1, ln-1):
			if sl[i] & 0x01 != current:
				current ^= 0x01
				self.footprint.append(cnt)
				cnt = 1
			else:
				cnt += 1
		self.footprint_len = len(self.footprint)
		self.slice_len = ln

class VerticalLeaf(object):

	def __init__(self, sl):
		self.slice = sl
		self.all_white = False
		self.stats()

	def stats(self):
		outcome = len(self.slice) - cv2.countNonZero(self.slice)
		self.all_white = outcome <= 5


class LeafStats(object):

	def __init__(self, leafs, vertical_leafs):
		self.leafs = leafs
		self.vleafs = vertical_leafs
		self.stats()

	def init_distribution(self):
		distribution = [0]*30
		return distribution

	def calc_open_hand_shape(self, distribution):
		all_leafs = sum(distribution)
		fingers = distribution[9] > 0.3*all_leafs or \
				  distribution[7] + distribution[9] > 0.3*all_leafs or \
				  sum(distribution[5:8]) > 0.3*all_leafs or \
				  sum(distribution[6:9]) > 0.3*all_leafs
		compact_body = distribution[3] > 0.1*all_leafs
		long_fingers = sum(distribution[5:14]) > 0.6*all_leafs
		if (fingers and compact_body) or long_fingers:
			return True
		return False

	def calc_half_face_shape(self, distribution):
		all_leafs = sum(distribution)
		half_face = distribution[2] > 0.85*all_leafs or \
					distribution[2] + distribution[1] > 0.85*all_leafs
		if half_face:
			return True
		return False

	def calc_cut_face_shape(self, distribution):
		all_leafs = sum(distribution)
		cut_face = distribution[2] > 0.4*all_leafs and \
				   (sum(distribution[1:4]) > 0.8*all_leafs)
		cut_face_other = distribution[2] > 0.4*all_leafs and \
					distribution[3] + distribution[4] < distribution[2] and \
					distribution[3] + distribution[3] > 0.3*all_leafs
		if cut_face or cut_face_other:
			return True
		return False

	def calc_ear_shape(self, distribution):
		"""
			thumb > 0.25
			ear < 0.25 ?
		"""
		all_leafs = sum(distribution)
		small_ear = all_leafs < 15 and distribution[4] > 1
		big_ear = (distribution[2] > 0.5*all_leafs and distribution[4] > 0.25*all_leafs) or \
		   		  (distribution[5] > 0.25*all_leafs and distribution[3] > 0.5*all_leafs)
		if small_ear or big_ear:
			return True
		return False

	def calc_half_hand_shape(self, distribution):
		all_leafs = sum(distribution)
		half_hand = sum(distribution[4:6]) > 0.5*all_leafs and distribution[4] > 0.3*all_leafs
		if half_hand:
			return True
		return False

	def calc_compact_shape(self, distribution):
		all_leafs = sum(distribution)
		compact_no_border = distribution[3] > 0.6*all_leafs
		compact_with_border = ((distribution[3] + distribution[2]) > 0.72*all_leafs and distribution[3] >= 0.4*all_leafs)
		compact_spread = len(set(distribution)) > 4 and distribution[3] >= 0.4*all_leafs
		if compact_no_border or compact_with_border or compact_spread:
			return True
		return False

	def calc_one_color_hand(self, distribution):
		if distribution[1] > sum(distribution) - 2:
			return True
		return False

	def calc_one_color_head(self, distribution):
		if distribution[1] == sum(distribution):
			return True
		return False

	def calc_edge_dev(self, edge):
		if edge:
			return np.std(edge)
		return 0

	def calc_thickness_dev(self, thickness):
		if len(thickness) > 4:
			s = sorted(thickness)
			s = s[3:]
			return np.std(s)
		return 0

	def calc_thickness_jumps(self, thickness):
		if len(thickness) > 4:
			return check_jumps(thickness)
		return -1

	def calc_line_variance(self, edge):
		if len(edge) > 5:
			s = sorted(edge)
			s = s[2:-2]
			length = len(s)
			low = s[0]
			high = s[-1]
			rng = high - low
			step = rng/length
			var = 0
			for i in range(0, length):
				theory = low + i*step
				real = s[i]
				diff = real - theory
				diff = diff**2
				var += diff
			return var/length
		return 0

	def calc_shape_type(self, thickness_dev, edge_dev):
		shape_type = "normal"
		if edge_dev > 40 and thickness_dev < 25:
			shape_type = "rotated hand"
		if edge_dev > 40 and thickness_dev > 40:
			shape_type = "blurred"
		return shape_type

	def stats(self):
		many_small_rects = False
		distribution = self.init_distribution()
		edge = []
		thickness = []

		for leaf in self.leafs:
			fst = leaf.first
			lst = leaf.last
			footprint = leaf.footprint
			footprint_len = leaf.footprint_len
			if fst == WHITE and footprint_len == 1 and footprint[0] > leaf.slice_len/2:
				trace = edge[-4:]
				if len(trace) > 2 and sum(trace) < 20:
					footprint.insert(0, 0)
			if footprint_len > 9:
				many_small_rects = True
			distribution[footprint_len+1] += 1
			if footprint_len > 0:
				edge.append(footprint[0])
			if footprint_len == 1:
				thickness.append(leaf.slice_len - footprint[0])
			if footprint_len == 2:
				thickness.append(footprint[1])

		up_down_white = False
		up_down_count = 0
		for leaf in self.vleafs:
			if leaf.all_white:
				up_down_count += 1
		if up_down_count >= 2:
			up_down_white = True

		self.thick_jumps = self.calc_thickness_jumps(thickness)
		self.thick_dev = self.calc_thickness_dev(thickness)
		self.deviation = self.calc_edge_dev(edge)

		self.shape_type = self.calc_shape_type(self.thick_dev, self.deviation)
		
		self.many_small_rects = many_small_rects
		self.ear_shape = self.calc_ear_shape(distribution)
		self.half_face_shape = self.calc_half_face_shape(distribution)
		self.cut_face_shape = self.calc_cut_face_shape(distribution)
		self.open_hand_shape = self.calc_open_hand_shape(distribution)
		self.half_hand_shape = self.calc_half_hand_shape(distribution)
		self.compact_shape = self.calc_compact_shape(distribution)
		self.up_down_white = up_down_white
		self.one_color = self.calc_one_color_head(distribution)
		self.edge_var = self.calc_line_variance(edge)
		#debuging process
		self.distribution = distribution
		self.edge = edge
		self.thickness = thickness

	def pretty_print(self):
		print "****************math***************"
		print "first edge dev:", self.deviation
		print "thickness dev:", self.thick_dev
		print "thickness jumps:", self.thick_jumps
		print "***************shapes**************"
		print "shape_type:", self.shape_type
		print "ear shape:", self.ear_shape
		print "open hand shape:", self.open_hand_shape
		print "compact shape:", self.compact_shape
		print "many small rects:", self.many_small_rects
		print "up down white:", self.up_down_white
		print "half face:", self.half_face_shape
		print "cut face:", self.cut_face_shape
		print "half hand:", self.half_hand_shape
		print "***********************************"
		print self.distribution
		print self.edge
		print self.thickness

class BodyPartsModel(object):

	def __init__(self, img):
		self.img = img
		self.h = self.img.shape[0]
		self.w = self.img.shape[1]
		self.outcome = UNKNOWN
		self.process()


	def has_not_enough_information(self):
		"""evaluate if img has enough information to distinguish
		   between hand and head"""
		img = self.img
		size = img.shape[0]*img.shape[1]
		if size < 4000:
			return True
		if size < 10000:
			nonZero = cv2.countNonZero(img)
			if nonZero > 0.95*size:
				return False
			if img.shape[0] > 1.5*img.shape[1] and nonZero > 0.3*size:
				return False
			return True
		if 10000 <= size < 40000:
			nonZero = cv2.countNonZero(img)
			if nonZero < 0.3*size:
				return True
		return False


	def get_leafs(self):
		img = self.img
		h = img.shape[0]
		amount = h/HORIZONTAL_JUMP
		fst_jumps = amount/2
		sec_jumps = amount/2
		c = 0
		leafs = []
		for i in range(0, fst_jumps):
			sl = img[c:c+1][0].copy()
			img[c:c+1][0] = 0
			c += HORIZONTAL_JUMP/2
			leafs.append(Leaf(sl))
		for i in range(0, sec_jumps):
			sl = img[c:c+1][0].copy()
			img[c:c+1][0] = 0
			c += HORIZONTAL_JUMP
			leafs.append(Leaf(sl))
		return leafs

	def get_vertical_leafs(self):
		img = self.img
		w = img.shape[1]
		how_many = int(0.75*w/VERTICAL_JUMP)
		amount = min(8, how_many)
		c = w/4
		leafs = []
		for i in range(0, amount):
			sl = img[:,c:c+1]
			c += VERTICAL_JUMP
			leafs.append(VerticalLeaf(sl))
		return leafs

	def process(self):
		if self.has_not_enough_information():
			return self.outcome
		vleafs = self.get_vertical_leafs()
		leafs = self.get_leafs()
		stats = LeafStats(leafs, vleafs)
		if stats.open_hand_shape and not stats.up_down_white:
			self.outcome = HAND
			print "open hand->HAND"
		elif stats.compact_shape and stats.up_down_white:
			self.outcome = HEAD
			print "compact&up down->FACE"
		elif stats.up_down_white and stats.ear_shape:
			self.outcome = HEAD
			print "up down&ear shape->FACE"
		elif stats.half_face_shape and stats.shape_type != "blurred":
			self.outcome = HEAD
			print "half face"
		elif stats.cut_face_shape and stats.up_down_white:
			self.outcome = HEAD
			print "cut face"
		elif stats.compact_shape and stats.ear_shape:
			self.outcome = HAND
			print "compact&ear shape"
		elif stats.half_hand_shape:
			self.outcome = HAND
			print "half hand"
		elif stats.many_small_rects:
			self.outcome = HAND
			print "small rects"
		elif stats.compact_shape and stats.thick_jumps > 1:
			self.outcome = HAND
			print "thick jumps hand"
		elif stats.compact_shape and stats.thick_jumps == 0:
			self.outcome = HEAD
			print "thick jumps head"
		elif 0 < stats.deviation < max(12, self.w*0.07):
			self.outcome = HEAD
			print "small dev", stats.deviation
		elif 0 < stats.deviation < 35 and stats.thick_dev < 30:
			self.outcome = HAND
			print "normal dev", stats.deviation, stats.thick_dev
		elif 0 < stats.edge_var < 144 and stats.thick_dev < 30:
			self.outcome = HAND
			print "edge variation"
		elif stats.deviation > 35 or stats.thick_dev > 30:
			self.outcome = HEAD_AND_HAND
			print "huge dev", stats.deviation, stats.thick_dev
		elif stats.one_color:
			self.outcome = HEAD
			print "one color head"
		else:
			self.outcome = HAND
			print "one color hand"
		self.stats = stats

	def get_value(self):
		return BODY_PARTS[self.outcome]


def save_to_file(results):
	f = open("C:\\Python27\\pdym\\imgs\\res.txt", 'w')
	for r in results:
		f.write(r)
		f.write("\n")
	f.close()


def read_from_file():
	results = []
	f = open("C:\\Python27\\pdym\\imgs\\res.txt", 'r')
	for line in f:
		save = line[:-1]
		results.append(save)
	return results


def compare_results(orig_results, new_results):
	if len(orig_results) == 0:
		return True
	i = 0
	bad = 0
	for r in orig_results:
		if r != new_results[i]:
			print i+1, "...original:", r, "...new:", new_results[i]
			bad += 1
		i += 1
	if bad == 0:
		return True
	return False

DEBUG = True
DEBUG = False
FORCE = False
def main():
	path = "C:\\Python27\\pdym\\imgs\\img%so.png"
	lst = range(3, 10)
	results = []
	for i in lst:
		p = path % i
		img = cv2.imread(p)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		b = BodyPartsModel(img)
		value = b.get_value()
		print "Img:", i, value
		if DEBUG:
			results.append(value)
		if not DEBUG:
			b.stats.pretty_print()
			cv2.imshow('Img%s' % i, img)
	if DEBUG:
		print "Comaparison:"
		if compare_results(read_from_file(), results):
			save_to_file(results)
			print "...OK"
		if FORCE:
			save_to_file(results)

main()
cv2.waitKey(0)