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
    get_roi


c = cv2.VideoCapture(0)
if cv2.__version__.find('2.4.8') > -1:
    # reading empty frame may be necessary 
    _, f = c.read()

def get_head_rect(img, cnt):
    """
    Taken from: calibration2
    """
    rect = list(cv2.boundingRect(cnt))
    rect[3] = rect[3]/2
    approx_roi = get_roi(img, rect)
    roi = approx_roi.copy()
    #cv2.imshow('roihead', roi)
    cnts, hier = cv2.findContours(roi, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = get_biggest_cnt(cnts)

    rect_inside = list(cv2.boundingRect(cnt))
    rect[0] = rect[0] + rect_inside[0]
    rect[1] = rect[1] + rect_inside[1]
    rect[2] = rect_inside[2]
    rect[3] = rect_inside[3]
    return rect


def find_important_planes(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    h_, s_, v_ = cv2.split(hsv)
    y, u, v = cv2.split(yuv)
    return h_, s_, v_, y, u, v

def hsv(img):
	return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def hvv(img):
	hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
	h_, s_, v_ = cv2.split(hsv)
	y, u, v = cv2.split(yuv)
	hvv = cv2.merge((h_, v, v_))
	return hvv

def discover_regions(planes):
    """
    Taken from: calibration2
    OTSU method is being used.
    """
    h_, v_, v = planes
    thr, thresholded = cv2.threshold(v_, 0, 255, cv2.THRESH_OTSU)
    
    cnts, hier = cv2.findContours(thresholded.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnt = get_biggest_cnt(cnts)
    if cnt is None:
        return None

    rect = get_head_rect(thresholded, cnt)
    head_mask = get_roi(thresholded, rect)
    return head_mask, rect

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

def hist(PLANE="V"):
	"""
	Creates 1d histograms of H, S planes from HSV 
	or U, V planes from YUV color space.
	Also creates 3d histograms of HS, UV, HV planes.
	2d histograms (as an image) can be created for HS and HV planes.
	Specify PLANE parameter for choose of histogram type.
	Examples:
	PLANE="H"
	PLANE="HS"
	PLANE="HSp" (for image version)
	"""
	cnt = 0
	values = []

	while(1):
		_,f = c.read()
		h_, s_, v_, y, u, v = find_important_planes(f)
		mask, rect = discover_regions((h_, v_, v))
		if PLANE == "H":
			roi_h = get_roi(h_, rect)
			values.extend(values_1d(roi_h, mask))
		elif PLANE == "V":
			roi_v = get_roi(v, rect)
			values.extend(values_1d(roi_v, mask))
		elif PLANE == "S":
			roi_s = get_roi(s_, rect)
			values.extend(values_1d(roi_s, mask))
		elif PLANE == "U":
			roi_u = get_roi(u, rect)
			values.extend(values_1d(roi_u, mask))
		elif PLANE == "HS":
			roi_h = get_roi(h_, rect)
			roi_s = get_roi(s_, rect)
			values.extend(values_2d(roi_h, roi_s, mask))
		elif PLANE == "UV":
			r1 = get_roi(u, rect)
			r2 = get_roi(v, rect)
			values.extend(values_2d(r1, r2, mask))
		elif PLANE == "HV":
			r1 = get_roi(h_, rect)
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
			PLANE = ""

		cv2.imshow('IMG', mask)
		k = cv2.waitKey(20)
		cnt += 1
		if cnt > 10:
			cv2.destroyAllWindows()
			c.release()
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

hist()

