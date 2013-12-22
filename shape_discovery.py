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

	def __init__(self, light="Night"):
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
		self.current_img = None
		self.hsv_range = [1, 2, 145, 200]
		self.threshold = 100

	def set_color_range(self, color_range):
		self.hsv_range = color_range

	def set_threshold(self, threshold):
		self.threshold = threshold

	def discover(self, img, rect=None):
		if rect == None:
			return
		x,y,w,h = rect
		roi = img[y:y+h, x:x+w]
		if self.light == "Day":
			roi_trf = self.apply_hsv_transformation(roi)
		else:
			roi_trf = self.apply_value_threshold_transformation(roi)
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

	def is_real(self):
		roi = self.current_img
		if roi is None:
			return False
		w, h = roi.shape
		all_pixels = 20*w
		part_of_roi = roi[h/4:3*h/4,:]
		amount = cv2.countNonZero(part_of_roi)
		if amount > 0.05*all_pixels:
			return True
		else:
			return False

	def smart_filter(self, hch, semi):
		h1 = cv2.inRange(hch, np.array([1],np.uint8), 
                       		 np.array([7],np.uint8))
		h2 = cv2.inRange(hch, np.array([145],np.uint8), 
                       		 np.array([200],np.uint8))
		hch = cv2.bitwise_or(h1, h2)
		hch = cv2.erode(hch, self.element)
		hch = cv2.erode(hch, self.element)
		hch = cv2.dilate(hch, self.element)
		h,w = hch.shape
		hp = 10; wp = 10
		for j in range(h/hp):
			for i in range(w/wp):
				part = hch[j*hp:(j+1)*hp, i*wp:(i+1)*wp]
				cnt = cv2.countNonZero(part)
				if cnt < 10:
					semi[j*hp:(j+1)*hp, i*wp:(i+1)*wp] = 0
		return semi

	def apply_hsv_transformation(self, roi):
		hsv1 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		h1,s1,v1 = cv2.split(hsv1)
		HSV_1_D = 120
		HSV_1_UP = 190
		HSV_2_D = 4
		HSV_2_UP = 20
		hdown = cv2.inRange(h1, np.array(self.hsv_range[0], np.uint8), 
                       		 np.array(self.hsv_range[1], np.uint8))
		hup = cv2.inRange(h1, np.array(self.hsv_range[2], np.uint8), 
                        	 np.array(self.hsv_range[3], np.uint8))
		h = cv2.bitwise_or(hdown, hup)
		h = cv2.dilate(h, self.element)
		h = cv2.dilate(h, self.element)
		h = cv2.medianBlur(h, 3)
		contours, hier = cv2.findContours(h, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		contour = self.biggest_cnt(contours)
		if contour == None:
			return None
		c1 = cv2.approxPolyDP(contour, 5, True)
		h1 = np.zeros(h.shape, np.uint8)
		cv2.drawContours(h1,[c1],-1,(255,0,0),-1)
		self.current_img = h1
		#cv2.imshow('h1', h1)
		return h1

	def apply_value_threshold_transformation(self, roi):
		hsv1 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		h1,s1,v1 = cv2.split(hsv1)
		dummy, v1 = cv2.threshold(v1, self.threshold, 255, cv2.THRESH_BINARY)
		c1, hier = cv2.findContours(v1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		c1 = self.biggest_cnt(c1)
		if c1 == None:
			return None
		c1 = cv2.approxPolyDP(c1, 5, True)
		v1 = np.zeros(v1.shape, np.uint8)
		cv2.drawContours(v1,[c1],-1,(255,0,0),-1)
		v1 = self.smart_filter(h1, v1)
		self.current_img = v1
		#cv2.imshow('h1', v1)
		return v1

	def _change_bgr_to_gray(self):
		path = "C:\\Python27\\pdym\\imgs\\img%s.png"
		path2 = "C:\\Python27\\pdym\\imgs\\img%so.png"
		try:
			for i in range(0, 189):
				img = cv2.imread(path % i)
				imgt = self.apply_value_threshold_transformation(img)
				cv2.imwrite(path2 % i, imgt)
		except Exception, a:
			print a

	def _change_bgr_to_hsv(self):
		path = "C:\\Python27\\pdym\\imgs\\img%s.png"
		path2 = "C:\\Python27\\pdym\\imgs\\img%so.png"
		try:
			for i in range(230, 291):
				img = cv2.imread(path % i)
				imgt = self.apply_hsv_transformation(img)
				cv2.imwrite(path2 % i, imgt)
		except Exception, a:
			print a

	def additional_informations(self):
		pass


def main1():
	sd = ShapeDiscovery()
	sd._change_bgr_to_gray()

def main2():
	path = "C:\\Python27\\pdym\\imgs\\img%s.png"
	p = path % 17
	img = cv2.imread(p)
	a = time.time()
	sd = ShapeDiscovery()
	value = sd.discover(img, None)
	print time.time() - a
	print value

#main1()
#main2()