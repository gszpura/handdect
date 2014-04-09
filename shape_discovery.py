import cv2
import numpy as np
import time

from main_utils import get_biggest_cnt, draw_circles
from body_model.body_model import BodyPartsModel

BODY_PARTS = ["UNKNOWN", "OPEN HAND", "ONE FINGER", "THUMB", "PALM", "FACE", "TWO FINGERS"]


def rev_area(defect):
    return 1/float(defect[0][3])

def _select_defect(defect, cnt, cnt_area, roi_height):
	s, e, f, a = defect[0]
	sp = tuple(cnt[s][0])
	ep = tuple(cnt[e][0])
	defect_level = max(sp[1], ep[1])
	if defect_level < 0.33*roi_height and \
		a > 0.33*cnt_area:
		return True
	return False

class ShapeDiscovery(object):

	def __init__(self):
		self.rects = []
		self.last = None
		self.img = None
		self.element = cv2.getStructuringElement(cv2. MORPH_CROSS,(3,3))
		self.debug = True
		self.rect1_score = 0
		self.rect2_score = 0

	def discover(self, img, rect=None):
		if rect == None:
			return
		x,y,w,h = rect
		roi = img[y:y+h, x:x+w]
		roi_trf = self.apply_approxing_transformation(roi)
		bpm = BodyPartsModel(roi_trf)
		shape_type = BODY_PARTS[bpm.get_value()]
		shape_type = self.defects_info(roi_trf, shape_type)
		self.last = shape_type
		return shape_type

	def apply_approxing_transformation(self, roi):
		cp = roi.copy()
		c1, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		c1 = get_biggest_cnt(c1)
		if c1 == None:
			return None
		c1 = cv2.approxPolyDP(c1, 5, True)
		v1 = np.zeros(roi.shape, np.uint8)
		cv2.drawContours(v1,[c1],-1,(255,0,0),-1)
		return v1

	def select_defects(self, defects, cnt, roi):
		out = sorted(defects, key=rev_area)
		cnt_area = cv2.moments(cnt)["m00"]
		out = [d[0][3] for d in out[0:5] if _select_defect(d, cnt, cnt_area, roi.shape[0])]
		return out

	def defects_info(self, roi, shape_cue):
		cp = roi.copy()
		cnts, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnt = get_biggest_cnt(cnts)
		hull = cv2.convexHull(cnt, returnPoints = False)
		defects = cv2.convexityDefects(cnt, hull)
		defs = self.select_defects(defects, cnt, roi)
		# cv2.imshow('approxing', roi)
		defects_count = len(defs)
		if shape_cue == BODY_PARTS[2]: #ONE FINGER
			return shape_cue
		elif shape_cue == BODY_PARTS[3]:
			return BODY_PARTS[3] #THUMB
		elif shape_cue == BODY_PARTS[6]:
			print "two-fing:body_model", defects_count
			return BODY_PARTS[6] #TWO FINGERS
		elif shape_cue == BODY_PARTS[1] and defects_count < 3:
			print "two-fing:defects"
			return BODY_PARTS[6] #TWO FINGERS
		elif shape_cue == BODY_PARTS[1]:
			return BODY_PARTS[1] #OPEN HAND
		elif shape_cue == BODY_PARTS[5]:
			return BODY_PARTS[4] #FACE -> PALM
		else:
			return shape_cue

def main2():
	path = "C:\\Python27\\pdym\\imgs\\img%s.png"
	p = path % 17
	img = cv2.imread(p)
	a = time.time()
	sd = ShapeDiscovery()
	value = sd.discover(img, None)
	print time.time() - a
	print value

#main2()