import cv2
import numpy as np

from config import CLASSIFIER
from main_utils import find_contours, \
	get_biggest_cnt, \
	fill_in_contour

class Cleaner(object):

	def __init__(self, light="Day"):
		self.element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
		self.light = light

	def clean_at_day(self, img):
		element = self.element
		d = cv2.erode(img, element)
		d = cv2.erode(d, element)
		d = cv2.dilate(d, element)
		return d

	def clean_at_night(self, img):
		element = self.element
		d = cv2.erode(img, element)
		d = cv2.dilate(d, element)
		return d

	def clean_linear(self, img):
		if self.light == "Night":
			return self.clean_at_night(img)
		elif self.light == "Day":
			return self.clean_at_day(img)

	def clean_bayes(self, img):
		element = self.element
		d = cv2.erode(img, element)
		d = cv2.erode(d, element)
		d = cv2.erode(d, element)
		d = cv2.erode(d, element)
		return d

	def clean(self, img):
		if CLASSIFIER == "bayes":
			return self.clean_bayes(img)
		elif CLASSIFIER == "linear":
			return self.clean_linear(img)

	def apply_dilates(self, roi):
		c = len(find_contours(roi))
		roi = cv2.dilate(roi, self.element)
		roi = cv2.dilate(roi, self.element)
		if CLASSIFIER == "bayes":
			roi = cv2.dilate(roi, self.element)
			roi = cv2.dilate(roi, self.element)
			roi = cv2.dilate(roi, self.element)
			roi = cv2.dilate(roi, self.element)
		if c > 12:
			r = cv2.dilate(roi, self.element)
			r = cv2.dilate(r, self.element)
			r = cv2.dilate(r, self.element)
			r = cv2.dilate(r, self.element)
			r = cv2.dilate(r, self.element)
			if c > 15:
				r = cv2.dilate(r, self.element)
				r = cv2.dilate(r, self.element)
			return r
		return roi

	def apply_lattice_cleaning(self, roi):
		h, w = roi.shape
		offset = int(0.6*h)
		y_times = 10
		x_times = 20
		x_l = w/x_times
		y_l = (h - offset)/y_times

		threshold_all = 0.2*x_l*y_l
		
		a = np.zeros([y_times, x_times], np.uint8)
		for w_s in xrange(x_times/2):
			for h_s in xrange(y_times):
				y_up = offset + h_s*y_l
				x_up = w_s*x_l

				x_down = w - w_s*x_l
				y_down = h - h_s*y_l
				r = roi[y_up:y_up+y_l, x_up:x_up+x_l]
				r_down = roi[y_down-y_l:y_down, x_down-x_l:x_down]
				
				cr = cv2.countNonZero(r)
				crd = cv2.countNonZero(r_down)
				
				if cr > threshold_all or \
				   a[max(0, h_s-1)][max(0, w_s)] + \
				   a[max(0, h_s)][max(0, w_s-1)] \
									== 2:
					a[h_s][w_s] = 1
					roi[y_up:y_up+y_l, x_up:x_up+x_l] = 255
				elif cr > 0:
					roi[y_up:y_up+y_l, x_up:x_up+x_l] = 0

				if crd > threshold_all or \
				   a[min(y_times - 1, y_times - h_s - 1)][min(x_times - 1, x_times - w_s)] + \
				   a[min(y_times - 1, y_times - h_s)][min(x_times - 1, x_times - w_s - 1)] \
									== 2:
					a[y_times - h_s - 1][x_times - w_s - 1] = 1
					roi[y_down-y_l:y_down, x_down-x_l:x_down] = 255
				elif cr > 0:
					roi[y_down-y_l:y_down, x_down-x_l:x_down] = 0
		return roi

	def apply_rebuild(self, roi):
		"""
		Takes ROI with hand or face, finds biggest contour
		and fills in holes with use of approxPolyDP and drawContours. 
		"""
		contours = find_contours(roi)
		cnts = get_biggest_cnt(contours, how_many=3)
		if cnts == None:
			return None
		cnts[0] = cv2.approxPolyDP(cnts[0], 5, True)
		rebuilded = np.zeros(roi.shape, np.uint8)
		if len(cnts) > 0: [fill_in_contour(rebuilded, cnt) for cnt in cnts]
		return rebuilded


