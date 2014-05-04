import cv2
import numpy as np
import time

from main_utils import get_biggest_cnt, \
	draw_circles, \
	draw_rects, \
	draw_info, \
	get_roi, \
	minimal_rect, \
	find_contours, \
	fill_in_contour
from body_model.body_model import BodyPartsModel

BODY_PARTS = ["UNKNOWN", "OPEN HAND", "ONE FINGER", "THUMB", "PALM", "FACE", "TWO FINGERS"]


def center_of_mass(cnt):
	m = cv2.moments(cnt)
	mass = (m["m10"]/m["m00"], m["m10"]/m["m00"])
	return mass

#TODO: finish
def reshape_rect_based_on_mass(roi, rect):
	c = find_contours(roi)
	cnt = get_biggest_cnt(find_contours(roi))
	if len(cnt) > 0:
		x, y = center_of_mass(cnt[0])
		xr, yr, wr, hr = rect
		print (x, y), (xr, yr)

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
		rect = minimal_rect(rect)
		roi = get_roi(img, rect)
		self.apply_dilates(roi)
		roi = self.apply_lattice_cleaning(roi)
		roi_trf = self.apply_rebuild(roi)

		#if rect[0] < 100:
		#	cv2.imshow('iii', roi_trf)
		bpm = BodyPartsModel(roi_trf)
		shape_type = BODY_PARTS[bpm.get_value()]
		shape_type = self.correct_shape_type(shape_type)
		#draw_info(img, shape_type, rect[0], rect[1])

		self.last = shape_type
		return shape_type

	def apply_dilates(self, roi):
		c = len(find_contours(roi))
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
		for w_s in range(x_times/2):
			for h_s in range(y_times):
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

	def correct_shape_type(self, shape_cue):
		if shape_cue == BODY_PARTS[5]: #FACE -> PALM
			return BODY_PARTS[4]
		else:
			return shape_cue

def test_main():
	path = "C:\\Python27\\pdym\\imgs\\img%s.png"
	p = path % 17
	img = cv2.imread(p)
	a = time.time()
	sd = ShapeDiscovery()
	value = sd.discover(img, None)
	print time.time() - a
	print value

#test_main()