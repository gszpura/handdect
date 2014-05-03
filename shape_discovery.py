import cv2
import numpy as np
import time

from main_utils import get_biggest_cnt, draw_circles
from body_model.body_model import BodyPartsModel

BODY_PARTS = ["UNKNOWN", "OPEN HAND", "ONE FINGER", "THUMB", "PALM", "FACE", "TWO FINGERS"]


def rev_area(defect):
    return 1/float(defect[0][3])

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
		#if roi_trf.shape[0] > 120:
		#	cv2.imshow('trf', roi_trf)
		bpm = BodyPartsModel(roi_trf)
		shape_type = BODY_PARTS[bpm.get_value()]
		shape_type = self.correct_shape_type(shape_type)
		self.last = shape_type
		return shape_type

	def apply_approxing_transformation(self, roi):
		"""
		Takes ROI with hand or face, finds biggest contour
		and fills in holes with use of approxPolyDP and drawContours. 
		"""
		contours, hierarchy = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		the_cnts = get_biggest_cnt(contours, how_many=3)
		if the_cnts == None:
			return None
		the_cnt = the_cnts[0]
		the_cnt = cv2.approxPolyDP(the_cnt, 5, True)

		rebuilded_roi = np.zeros(roi.shape, np.uint8)
		cv2.drawContours(rebuilded_roi, [the_cnt], -1, (255,0,0), -1)
		for cnt in the_cnts[1:]:
			cv2.drawContours(rebuilded_roi, [cnt], -1, (255,0,0), -1)
		return rebuilded_roi

	def correct_shape_type(self, shape_cue):
		if shape_cue == BODY_PARTS[5]: #FACE -> PALM
			return BODY_PARTS[4]
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