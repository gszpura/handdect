import cv2
import numpy as np
import time
from collections import deque
from hand_filter import HandFilter

WHITE = 1
BLACK = 0

HAND = 1
HEAD = 2
HEAD_AND_HAND = 3
UNKNOWN = 0
BODY_PARTS = ["UNKNOWN", "HAND", "HEAD", "HEAD_AND_HAND"]

HORIZONTAL_JUMP = 10
VERTICAL_JUMP = 30

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
		fingers = distribution[9]  + distribution[5] + distribution[7]
		if (distribution[9] >= 3 or distribution[5] >= 3  or distribution[7] >= 3) and fingers > 5 and distribution[3] > 3:
			return True
		return False

	def calc_ear_shape(self, distribution):
		if (distribution[2] > 9 and distribution[4] > 3) or (distribution[5] > 3 and not distribution[9] > 3):
			return True
		return False

	def calc_compact_shape(self, distribution):
		all_leafs = sum(distribution)
		if distribution[3] > 0.6*all_leafs or ((distribution[3] + distribution[2]) > 0.8*all_leafs and distribution[3] >= 0.4*all_leafs):
			return True
		return False

	def calc_one_color(self, distribution):
		if distribution[1] > sum(distribution) - 2:
			return True
		return False

	def stats(self):
		many_small_rects = False
		distribution = self.init_distribution()
		edge = []
		right_edge_trace = []
		left_edge_trace = []
		
		for leaf in self.leafs:
			fst = leaf.first
			lst = leaf.last
			left_edge_trace.append(fst)
			right_edge_trace.append(lst)
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

		up_down_white = False
		for leaf in self.vleafs:
			if leaf.all_white:
				up_down_white = True

		if edge:
			self.median = np.median(edge)
			self.deviation = np.std(edge)
		else:
			self.median = -1
			self.deviation = 0

		leaf_len = len(self.leafs)
		left_edge_white = sum(left_edge_trace)
		right_edge_white = sum(right_edge_trace)
		left_edge_white = left_edge_white > 0.9*leaf_len or left_edge_white >= leaf_len - 2
		right_edge_white = right_edge_white > 0.9*leaf_len or right_edge_white >= leaf_len - 2
		self.white_edge = left_edge_white or right_edge_white
		
		self.many_small_rects = many_small_rects
		self.ear_shape = self.calc_ear_shape(distribution)
		self.open_hand_shape = self.calc_open_hand_shape(distribution)
		self.compact_shape = self.calc_compact_shape(distribution)
		self.up_down_white = up_down_white
		self.one_color = self.calc_one_color(distribution)
		#debuging process
		self.distribution = distribution
		self.edge = edge

	def pretty_print(self):
		print "****************math***************"
		print "first edge position median:", self.median
		print "first edge position dev:", self.deviation
		print "***************shapes**************"
		print "ear shape:", self.ear_shape
		print "open hand shape:", self.open_hand_shape
		print "compact shape:", self.compact_shape
		print "************white color************"
		print "white edge:", self.white_edge
		print "many small rects:", self.many_small_rects
		print "up down white:", self.up_down_white

class BodyPartsModel(object):

	def __init__(self, img):
		self.img = img
		self.outcome = UNKNOWN
		self.process()


	def has_not_enough_information(self):
		"""evaluate if img has enough information to distinguish
		   between hand and head"""
		img = self.img
		size = img.shape[0]*img.shape[1]
		if size < 10000:
			return True
		if size < 40000:
			nonZero = cv2.countNonZero(img)
			if nonZero < 0.3*size:
				return True
		return False


	def get_leafs(self):
		img = self.img
		h = img.shape[0]
		amount = h/HORIZONTAL_JUMP
		fst_jumps = amount/2
		tmp_sec = amount/2 - 3
		sec_jumps = tmp_sec > 2 and tmp_sec or fst_jumps
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
		amount = w/VERTICAL_JUMP
		c = 20
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
		if stats.open_hand_shape:
			self.outcome = HAND
			print "open hand"
		elif stats.compact_shape and stats.up_down_white:
			self.outcome = HEAD
			print "compact&up_down"
		elif stats.up_down_white and stats.ear_shape:
			self.outcome = HEAD
			print "up_down&ear_shape"
		elif stats.compact_shape and stats.ear_shape:
			self.outcome = HAND
			print "compact&ear"
		elif stats.many_small_rects:
			self.outcome = HAND
			print "small rects"
		elif 0 < stats.deviation < 12:
			self.outcome = HEAD
			print "small_dev", stats.deviation
		elif 0 < stats.deviation < 35:
			self.outcome = HAND
			print "normal dev", stats.deviation
		elif stats.deviation > 35:
			self.outcome = HEAD_AND_HAND
			print "huge dev", stats.deviation
		elif stats.one_color and stats.white_edge:
			self.outcome = HAND
			print "one_color"
		self.stats = stats

	def get_value(self):
		return BODY_PARTS[self.outcome]


test_list = [None, None, None, None, None, None, None, None, None, None]
test_list.extend(["HEAD", "HAND", "HEAD", "HAND", "HEAD", "HAND", "HEAD_AND_HAND", "HEAD", "HEAD_AND_HAND", "HEAD"])

def main():
	path = "C:\\Python27\\pdym\\imgs\\img%so.png"
	for i in range(12,13):
		p = path % i
		img = cv2.imread(p)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		b = BodyPartsModel(img)
		print "Img:", i
		if b.get_value() != test_list[i]:
			print "############################"
			print b.get_value(), "should be:", test_list[i]
			print "############################"
			b.stats.pretty_print()
			cv2.imshow('Img%s' % i, img)
		print "\n"
		

main()
cv2.waitKey(0)