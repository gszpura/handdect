import cv2
import numpy as np
import time

from main_utils import draw_circles

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


class HandFilter(object):

	def __init__(self, light="Daylight"):
		self.rects = []
		self.last = None
		self.img = None
		self.element = cv2.getStructuringElement(cv2. MORPH_CROSS,(3,3))
		self.debug = True
		self.rect1_score = 0
		self.rect2_score = 0
		self.light = light

	def update(self, img, last, rects):
		self.rects = rects
		self.last = last
		if len(rects) == 0:
			return
		if len(rects) == 1:
			#check if head is there
			r1 = rects[0]
			x1,y1,w1,h1 = r1
		if len(rects) == 2:
			r1 = rects[0]
			r2 = rects[1]
			x1,y1,w1,h1 = r1
			x2,y2,w2,h2 = r2
			roi1 = img[y1:y1+h1, x1:x1+w1]
			roi2 = img[y2:y2+h2, x2:x2+w2]
			if self.light == "Daylight":
				#self.hsv(roi1, roi2)
				self.hsv_alt(roi1, roi2)
			else:
				self.gray(roi1, roi2)

	def biggest_cnt(self, cnts):
		biggest = None
		biggest_area = 0
		for cnt in cnts:
			m = cv2.moments(cnt)
			if m["m00"] > biggest_area:
				biggest = cnt
				biggest_area = m["m00"]
		return biggest

	def center_of_mass(self, cnt):
		m = cv2.moments(cnt)
		try:
			x = int(m["m10"]/m["m00"])
			y = int(m["m01"]/m["m00"])
		except ZeroDivisionError:
			x = -1
			y = -1
		return x, y 

	def radius(self, cnt):
		rect = cv2.boundingRect(cnt)
		return int(rect[2]/float(2))

	def circle_roi(self, shape, pos, r):
		img_h, img_w = shape
		img = np.zeros((img_h, img_w), np.uint8)
		cv2.circle(img, pos, r, 255, -1)
		return img

	def hsv(self, roi1, roi2):
		"""finds contours with use of hsv color space"""
		hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
		hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
		if self.debug:
			cv2.imshow('orig1', hsv1)
			cv2.imshow('orig2', hsv2)
		h1,s1,v1 = cv2.split(hsv1)
		h2,s2,v2 = cv2.split(hsv2)
		HSV_1_D = 145
		HSV_1_UP = 200
		HSV_2_D = 4
		HSV_2_UP = 20
		hx = cv2.inRange(h1, np.array([HSV_1_D],np.uint8), 
                       		 np.array([HSV_1_UP],np.uint8))
		hy = cv2.inRange(h1, np.array([HSV_2_D],np.uint8), 
                        	 np.array([HSV_2_UP],np.uint8))
		h1 = cv2.bitwise_or(hx, hy)
		h1 = cv2.dilate(h1, self.element)
		h1 = cv2.dilate(h1, self.element)
		h1 = cv2.medianBlur(h1, 3)

		hx = cv2.inRange(h2, np.array([HSV_1_D],np.uint8), 
                       		 np.array([HSV_1_UP],np.uint8))
		hy = cv2.inRange(h2, np.array([HSV_2_D],np.uint8), 
                        	 np.array([HSV_2_UP],np.uint8))
		h2 = cv2.bitwise_or(hx, hy)
		
		h2 = cv2.dilate(h2, self.element)
		h2 = cv2.dilate(h2, self.element)
		h2 = cv2.medianBlur(h2, 3)
		if self.debug:
			cv2.imshow('contours1o', h1)
			cv2.imshow('contours2o', h2)

		c1, hier = cv2.findContours(h1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		c2, hier = cv2.findContours(h2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		c1 = self.biggest_cnt(c1)
		c2 = self.biggest_cnt(c2)
		self.c2 = cv2.approxPolyDP(c2, 25, True)
		self.c1 = cv2.approxPolyDP(c1, 25, True)
		h1 = np.zeros(h1.shape, np.uint8)
		h2 = np.zeros(h2.shape, np.uint8)
		cv2.drawContours(h2,[self.c2],-1,(255,0,0),-1)
		cv2.drawContours(h1,[self.c1],-1,(255,0,0),-1)
		if self.debug:
			cv2.imshow('contours1', h1)
			cv2.imshow('contours2', h2)
		self.h1 = h1
		self.h2 = h2
		self.bounding_circle_cue()

	def hsv_to_thresh(self, roi):
		h = roi.shape[0]
		w = roi.shape[1]
		ret = np.zeros(roi.shape, np.uint8)
		for w1 in range(0, w):
			for h1 in range(0, h):
				p = roi[h1][w1]
				if p[2] > 120:
					ret[h1][w1] = 255
		return ret

	def hsv_alt(self, roi1, roi2):
		"""finds contours with use of hsv color space"""
		hsv1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2HSV)
		hsv2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2HSV)
		h1,s1,v1 = cv2.split(hsv1)
		h2,s2,v2 = cv2.split(hsv2)
		#v1 = cv2.erode(v1, self.element)
		#v2 = cv2.erode(v2, self.element)
		dummy, v1 = cv2.threshold(v1, 110, 255, cv2.THRESH_BINARY)
		dummy, v2 = cv2.threshold(v2, 110, 255, cv2.THRESH_BINARY)

		c1, hier = cv2.findContours(v1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		c2, hier = cv2.findContours(v2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		c1 = self.biggest_cnt(c1)
		c2 = self.biggest_cnt(c2)
		self.c2 = cv2.approxPolyDP(c2, 5, True)
		self.c1 = cv2.approxPolyDP(c1, 5, True)

		v1 = np.zeros(v1.shape, np.uint8)
		v2 = np.zeros(v2.shape, np.uint8)
		cv2.drawContours(v2,[self.c2],-1,(255,0,0),-1)
		cv2.drawContours(v1,[self.c1],-1,(255,0,0),-1)
		
		if self.debug:
			cv2.imshow('orig1', v1)
			cv2.imshow('orig2', v2)

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
		
	def apply_hsv_alt_transformation(self, roi):
		hsv1 = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
		h1,s1,v1 = cv2.split(hsv1)
		dummy, v1 = cv2.threshold(v1, 90, 255, cv2.THRESH_BINARY)
		c1, hier = cv2.findContours(v1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
		c1 = self.biggest_cnt(c1)
		c1 = cv2.approxPolyDP(c1, 5, True)
		v1 = np.zeros(v1.shape, np.uint8)
		cv2.drawContours(v1,[c1],-1,(255,0,0),-1)
		v1 = self.smart_filter(h1, v1)
		return v1

	def _change_bgr_to_gray(self):
		path = "C:\\Python27\\pdym\\imgs\\img%s.png"
		path2 = "C:\\Python27\\pdym\\imgs\\img%so.png"
		try:
			for i in range(10, 180):
				img = cv2.imread(path % i)
				imgt = self.apply_hsv_alt_transformation(img)
				cv2.imwrite(path2 % i, imgt)
		except Exception, a:
			print a

	def bounding_circle_cue(self):
		mass1 = self.center_of_mass(self.c1)
		mass2 = self.center_of_mass(self.c2)
		radius1 = self.radius(self.c1)
		radius2 = self.radius(self.c2)
		croi1 = self.circle_roi(self.h1.shape, mass1, radius1)
		croi2 = self.circle_roi(self.h2.shape, mass2, radius2)
		
		h2 = cv2.bitwise_and(self.h2, croi2)
		h1 = cv2.bitwise_and(self.h1, croi1)
		count2 = cv2.countNonZero(h2)
		croi_count2 = cv2.countNonZero(croi2)
		croi_count1 = cv2.countNonZero(croi1)
		count1 = cv2.countNonZero(h1)
		ratio1 = count1/float(croi_count1)
		ratio2 = count2/float(croi_count2)
		if ratio1 > ratio2:
			self.rect2_score += 1
		elif ratio2 > ratio1:
			self.rect1_score += 1
		if self.debug:
			cv2.imshow('hsv1', h1)
			cv2.imshow('hsv2', h2)

	def convex_cue(self):
		pass

	def gray(self, roi1, roi2):
		g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
		g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
		dummy, g1 = cv2.threshold(g1, 100, 255, cv2.THRESH_BINARY)
		dummy, g1 = cv2.threshold(g2, 100, 255, cv2.THRESH_BINARY)
		cv2.imshow('gray1', g1)
		cv2.imshow('gray2', g2)

	def get_found_rect(self):
		if len(self.rects) == 1:
			return self.rects[0]
		elif len(self.rects) == 2:
			if self.rect1_score > self.rect2_score:
				print "rect1"
				return self.rects[1]
			elif self.rect2_score > self.rect1_score:
				print "rect2"
				return self.rects[0]
			else:
				return self.last
		elif len(self.rects) == 0:
			return self.last
