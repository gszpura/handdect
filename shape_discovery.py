import cv2
import numpy as np
import time

from body_model import BodyPartsModel

FIST = 0
ONE_FINGER = 1
TWO_FINGERS = 2
THREE_FINGERS = 3
FOUR_FINGERS = 4
PALM = 5



class HandInfo(object):

	def __init__(self):
		self.position = (-1, -1)
		self.gesture_type = -1
		self.thumb = -1


class ShapeDiscovery(object):

	def __init__(self, light):
		self.rects = []
		self.last = None
		self.img = None
		self.element = cv2.getStructuringElement(cv2. MORPH_CROSS,(3,3))
		self.debug = True
		self.rect1_score = 0
		self.rect2_score = 0
		if light == "DayDim":
			light = "Day"
		self.light = light
		self.threshold = 100

	def set_threshold(self, threshold):
		self.threshold = threshold

	def discover(self, img, rect=None):
		if rect == None:
			return
		x,y,w,h = rect
		roi = img[y:y+h, x:x+w]
		roi_trf = self.apply_approxing_transformation(roi)
		bpm = BodyPartsModel(roi_trf)
		return bpm.get_value()

	def biggest_cnt(self, cnts):
		biggest = None
		biggest_area = 0
		for cnt in cnts:
			m = cv2.moments(cnt)
			if m["m00"] > biggest_area:
				biggest = cnt
				biggest_area = m["m00"]
		return biggest

	def apply_approxing_transformation(self, roi):
		cp = roi.copy()
		c1, hier = cv2.findContours(cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		c1 = self.biggest_cnt(c1)
		if c1 == None:
			return None
		c1 = cv2.approxPolyDP(c1, 5, True)
		v1 = np.zeros(roi.shape, np.uint8)
		cv2.drawContours(v1,[c1],-1,(255,0,0),-1)
		#cv2.imshow('approxing', v1)
		return v1

	def additional_informations(self):
		pass

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