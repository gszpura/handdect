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
from cleaner import Cleaner

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

def reshape_if_above(img, rect):
	rect_above = [rect[0], max(0, rect[1]-rect[3]/2),
				 rect[2], min(rect[3]/2, rect[1])]
	roi = get_roi(img, rect_above)
	size = rect[2]*rect[3]

	if 0.3*size > cv2.countNonZero(roi) > 0.025*size:
		return [rect[0], rect_above[1], rect[2], rect[3] + rect_above[3]]
	return rect

class ShapeDiscovery(object):

	def __init__(self):
		self.last = None
		self.cleaner = Cleaner()

	def discover(self, img, rect=None):
		if rect == None:
			return
		if self.last == "PALM":
			rect = reshape_if_above(img, rect)

		roi = get_roi(img, rect)
		roi = self.cleaner.apply_dilates(roi)
		roi = self.cleaner.apply_lattice_cleaning(roi)
		roi_trf = self.cleaner.apply_rebuild(roi)

		bpm = BodyPartsModel(roi_trf)
		shape_type = BODY_PARTS[bpm.get_value()]
		shape_type = self.correct_shape_type(shape_type)
		#draw_info(img, shape_type, rect[0], rect[1])

		self.last = shape_type
		return shape_type

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