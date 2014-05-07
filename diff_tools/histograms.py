#-*- coding: utf-8 -*-
import cv2
import sys
import os
import random
import time

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pdym.main_utils import get_biggest_cnt, \
	draw_rects, \
    get_roi, \
    find_contours, \
    split_into_planes, \
    get_head_rect, \
    find_head_with_otsu, \
    init_camera, \
    release_camera


c = init_camera()

####Configuration####
# Choose from: H, S, U, V, HS, UV, HV, HSp, HVp
PLANE = "H"
#####################


def hsv(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def hvv(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	h_, s_, v_ = cv2.split(hsv)
	y, u, v = cv2.split(yuv)
	hvv = cv2.merge((h_, v, v_))
	return hvv


def values_1d(roi, mask):
	h, w = roi.shape
	v = []
	for i in range(0, h):
		for j in range(0, w):
			if mask[i][j] > 0:
				v.append(roi[i][j])
	return v


def values_2d(plane1, plane2, mask):
	h, w = plane1.shape
	v = []
	for i in range(0, h):
		for j in range(0, w):
			if mask[i][j] > 0:
				v.append((plane1[i][j], plane2[i][j]))
	return v


def hist():
	"""
	Creates 1d histograms of H, S planes from HSV 
	or U, V planes from YUV color space.
	Also creates 3d histograms of HS, UV, HV planes.
	2d histograms (as an image) can be created for HS and HV planes.
	"""
	cnt = 0
	values = []

	while(1):
		_,f = c.read()
		H, S, V, y, u, v = split_into_planes(f)
		mask, rect = find_head_with_otsu(V)
		if PLANE == "H":
			roi_h = get_roi(H, rect)
			values.extend(values_1d(roi_h, mask))
		elif PLANE == "V":
			roi_v = get_roi(v, rect)
			values.extend(values_1d(roi_v, mask))
		elif PLANE == "S":
			roi_s = get_roi(S, rect)
			values.extend(values_1d(roi_s, mask))
		elif PLANE == "U":
			roi_u = get_roi(u, rect)
			values.extend(values_1d(roi_u, mask))
		elif PLANE == "HS":
			roi_h = get_roi(H, rect)
			roi_s = get_roi(S, rect)
			values.extend(values_2d(roi_h, roi_s, mask))
		elif PLANE == "UV":
			r1 = get_roi(u, rect)
			r2 = get_roi(v, rect)
			values.extend(values_2d(r1, r2, mask))
		elif PLANE == "HV":
			r1 = get_roi(H, rect)
			r2 = get_roi(v, rect)
			values.extend(values_2d(r1, r2, mask))
		elif PLANE == "HSp":
			img = hsv(f)
			hist2d(get_roi(img, rect), mask)
			k = cv2.waitKey(0)
			break
		elif PLANE == "HVp":
			img = hvv(f)
			hist2d(get_roi(img, rect), mask)
			k = cv2.waitKey(0)
			break
		else:
			return

		cv2.imshow('IMG', mask)
		k = cv2.waitKey(20)
		cnt += 1
		if cnt > 10:
			break

	if len(PLANE) == 1:
		hist1d(values, PLANE)	
	if len(PLANE) == 2:
		p1, p2 = zip(*values)
		hist3d(p1, p2, PLANE)


def hist1d(values, name):
	n, bins, patches = plt.hist(values, 128, normed=0, facecolor='green', alpha=0.5)
	plt.xlabel(name)
	plt.show()


def hist2d(roi, mask):
	"""
	Histogram on the plane.
	Value is showed by means of color.
	"""
	hbins = 36
	sbins = 50
	hist = cv2.calcHist([roi], [0, 1], mask, histSize=[hbins, sbins], ranges=[0, 180, 0, 255])
	cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
	hist=np.int32(np.around(hist))
	histSize=[hbins, sbins]
	SQ = 5
	im2 = np.zeros((hbins*SQ, sbins*SQ, 1))
	for i in range(0, hbins):
	    for j in range(0, sbins):
	        v = hist[i, j]
	        for k in range(i*SQ, i*SQ+SQ):
	            for l in range(j*SQ, j*SQ+SQ):
	                im2[k,l] = v
	im2 = np.uint8(im2)
	cv2.imshow('IMG',im2)


def hist3d(plane1, plane2, names):
	bins = 32
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	hist, x, y = np.histogram2d(plane1, plane2, bins=bins)
	elements = (len(x) - 1) * (len(y) - 1)
	xpos, ypos = np.meshgrid(x[:-1]-40, y[:-1]+20)
	xpos = xpos.flatten()
	ypos = ypos.flatten()
	zpos = np.zeros(elements)

	dx = 0.5 * np.ones_like(zpos)
	dy = dx.copy()
	dz = hist.flatten()

	ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='r', zsort='average')
	plt.xlabel(names[0])
	plt.ylabel(names[1])
	plt.show()


if __name__ == "__main__":
	hist()
	release_camera(c)
