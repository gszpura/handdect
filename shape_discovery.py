import cv2
import numpy as np
import time

from main_utils import get_biggest_cnt, draw_circles, draw_rects, get_roi
from body_model.body_model import BodyPartsModel

BODY_PARTS = ["UNKNOWN", "OPEN HAND", "ONE FINGER", "THUMB", "PALM", "FACE", "TWO FINGERS"]

#TODO: move to main_utils
def find_contours(roi):
	contours, hierarchy = cv2.findContours(roi.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	return contours	

def fill_in_contour(roi, cnt):
	cv2.drawContours(roi, [cnt], -1, (255,0,0), -1)

def minimal_rect(rect):
	"""
	If rect is too small make it a bit bigger
	in ordert to not to miss part of the hand.
	"""
	x, y, w, h = rect
	if w < 70:
		x = max(0, x - 70)
		w = min(640, w + 140)
	if h < 70:
		y = max(0, y - 70)
		h = min(480, h + 140)
	if w > h:
		y = max(0, y - 40)
		h = min(480, h + 40)
	return [x, y, w, h]

def center_of_mass(cnt):
	m = cv2.moments(cnt)
	mass = (m["m10"]/m["m00"], m["m10"]/m["m00"])
	return mass

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
		#reshape_rect_based_on_mass(roi, rect)
		roi = self.apply_lattice_cleaning(roi)
		roi_trf = self.apply_rebuild(roi)

		bpm = BodyPartsModel(roi_trf)
		shape_type = BODY_PARTS[bpm.get_value()]
		shape_type = self.correct_shape_type(shape_type)
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
		cnts = get_biggest_cnt(contours, how_many=2)
		if cnts == None:
			return None
		the_cnt = cnts[0]
		the_cnt = cv2.approxPolyDP(the_cnt, 5, True)
		fill_in_contour(roi, the_cnt)
		if len(cnts) == 2: fill_in_contour(roi, cnts[1])
		return roi

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