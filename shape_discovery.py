import cv2
import numpy as np
import time

from main_utils import get_biggest_cnt
from body_model.body_model import BodyPartsModel

BODY_PARTS = ["UNKNOWN", "OPEN HAND", "ONE FINGER", "THUMB", "PALM", "FACE", "FACE & HAND", "TWO FINGERS"]


def rev_area(area):
    return 1/float(area)

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
		#cv2.imshow('approxing', v1)
		return v1

	def defects_info(self, roi, shape_cue):
		cp = roi.copy()
		cnts, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		cnt = get_biggest_cnt(cnts)
		hull = cv2.convexHull(cnt, returnPoints = False)
		defects = cv2.convexityDefects(cnt, hull)
		# display
		"""for i in range(defects.shape[0]):
		    s,e,f,d = defects[i,0]
		    if d < 6500:
		    	continue
		    start = tuple(cnt[s][0])
		    end = tuple(cnt[e][0])
		    far = tuple(cnt[f][0])
		    cv2.line(roi,start,end,[100,100,0],2)
		    cv2.circle(roi,far,5,[100,100,0],-1) 
		cv2.imshow('Def', roi) """
		
		biggest = [d[0][3] for d in defects]
		biggest2 = sorted(biggest, key=rev_area)
		biggest2 = biggest2[0:5]
		biggest2 = [area for area in biggest2 if area > 6000]
		defects_count = len(biggest2)
		if shape_cue == "ONE FINGER":
			return shape_cue
		if shape_cue == BODY_PARTS[1] and defects_count in (1, 2):
			return BODY_PARTS[3] #THUMB
		elif defects_count == 3:
			return BODY_PARTS[7] #TWO FINGERS
		elif shape_cue == BODY_PARTS[1] and defects_count > 3:
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