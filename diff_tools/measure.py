import numpy as np
import os
import time
import cv2
from matplotlib import pyplot as plt
import sys

MODULE = sys.modules[__name__]

PHOTO_DIR = "\\test_images\\"
PATH = os.path.dirname(__file__) + PHOTO_DIR

def measure(func):
	def decorate(**kwargs):
		start = time.time()
		res = func(**kwargs)
		stop = time.time()
		print "Overall:", stop - start
		return res
	return decorate


def harris_corner_measure():
	PHOTO = os.path.dirname(__file__) + "\\capture.jpg"
	img = cv2.imread(PHOTO)

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	start = time.time()
	gray = np.float32(gray)

	dst = cv2.cornerHarris(gray, 2, 3, 0.04)
	s2 = time.time()
	print "Harris corner detection:", s2 - start
	dst = cv2.dilate(dst, None)
	img[dst>0.02*dst.max()] = [0,0,255]
	stop = time.time()
	print "Overall time:", stop - start

	cv2.imshow('dst',img)
	if cv2.waitKey(0) & 0xff == 27:
	    cv2.destroyAllWindows()


def draw_points(img, points):
	for point in points:
		cv2.circle(img, tuple(point), 10, (255,0,0))


@measure
def sift_surf_measure(queryImage=PATH + 'back_hand0.jpg',
				      trainImage=PATH + 'back_scene3.jpg',
				      show_results=False,
				      operation=cv2.SIFT):
	img1 = cv2.imread(queryImage, 0)      
	img2 = cv2.imread(trainImage, 0)
	s1 = time.time()
	s = operation()

	kp1, des1 = s.detectAndCompute(img1, None)
	kp2, des2 = s.detectAndCompute(img2, None)

	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=20)

	f = cv2.FlannBasedMatcher(index_params, search_params)
	matches = f.knnMatch(des1, des2, k=2)

	good = []
	for m,n in matches:
	    if m.distance < 0.8*n.distance:
	        good.append(m)

	# summary
	s2 = time.time()
	print "Time:", s2 - s1
	if len(good) > 10:
		print "GOOD:", len(good)
	else:
		print "NOT THAT GOOD"

	if show_results:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])

		draw_points(img1, src_pts)
		draw_points(img2, dst_pts)

		cv2.imshow('src',img1)
		cv2.imshow('dst',img2)
		if cv2.waitKey(0) & 0xff == 27:
		    cv2.destroyAllWindows()


@measure
def orb_measure(queryImage=PATH + 'back_hand0.jpg',
				trainImage=PATH + 'back_scene11.jpg',
				show_results=False):
	img1 = cv2.imread(queryImage, 0)      
	img2 = cv2.imread(trainImage, 0)
	s1 = time.time()
	s = cv2.ORB()

	kp1, des1 = s.detectAndCompute(img1, None)
	kp2, des2 = s.detectAndCompute(img2, None)

	f = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = f.match(des1, des2)
	good = matches
	# summary
	s2 = time.time()
	print "Time:", s2 - s1
	if len(good) > 10:
		print "GOOD:", len(good)
	else:
		print "NOT THAT GOOD"

	if show_results:
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ])
		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ])

		draw_points(img1, src_pts)
		draw_points(img2, dst_pts)

		cv2.imshow('src',img1)
		cv2.imshow('dst',img2)
		if cv2.waitKey(0) & 0xff == 27:
		    cv2.destroyAllWindows()


def face_light(name):
	if name.find("back") > -1 or \
	   name.find("diff") > -1 or \
	   name.find("lamp") > -1:
	   return True
	return False


def validation_surf(matches):
	good = []
	for m,n in matches:
	    if m.distance < 0.8*n.distance:
	        good.append(m)
	return good


def validation_sift(matches):
	good = []
	for m,n in matches:
	    if m.distance < 0.7*n.distance:
	        good.append(m)
	return good


def validation_orb(matches):
	return matches


def flann_matcher():
	# prepare matcher
	FLANN_INDEX_KDTREE = 0
	index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
	search_params = dict(checks=50)
	f = cv2.FlannBasedMatcher(index_params, search_params)
	return f


def bf_matcher_orb():
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	return bf


def bf_matcher():
	bf = cv2.BFMatcher()
	return bf


def apply_operation(hand, scene, operation, validation, matcher):
	img1 = cv2.imread(hand, 0)
	img2 = cv2.imread(scene, 0)

	# initialize surf
	s = operation()
	kp1, des1 = s.detectAndCompute(img1, None)

	f = matcher()

	s1 = time.time()
	kp2, des2 = s.detectAndCompute(img2, None)
	s2 = time.time()
	if f.__class__.__name__ == 'FlannBasedMatcher':
		matches = f.knnMatch(des1, des2, k=2)
	elif f.__class__.__name__ == 'BFMatcher':
		matches = f.match(des1, des2)
	else:
		raise Exception("Wrong matcher")
	s3 = time.time()
	good = validation(matches)

	matchTime = s3 - s2
	detectTime = s2 - s1
	return len(good) > 15, detectTime, matchTime


def apply_surf(hand, scene):
	return apply_operation(hand, scene, cv2.SURF, validation_surf, flann_matcher)


def apply_sift(hand, scene):
	return apply_operation(hand, scene, cv2.SIFT, validation_sift, flann_matcher)


def apply_orb(hand, scene):
	return apply_operation(hand, scene, cv2.ORB, validation_orb, bf_matcher_orb)


def measure_test_set(operation='surf'):
	apply_function = getattr(MODULE, 'apply_' + operation)

	dir_ = os.path.dirname(__file__) + PHOTO_DIR
	ls = os.listdir(dir_)
	# filter out "gesture" group
	ls = [name for name in ls if not name.find("gesture") > -1]

	uplight = [name for name in ls if name.find("uplight") > -1]
	facelight = [name for name in ls if face_light(name)]
	daylight = []
	rotated = [name for name in ls if name.find("rotated") > -1]

	groups = [uplight, facelight, daylight, rotated]
	names = ["uplight", "facelight", "daylight", "rotated"]
	for i, group in enumerate(groups):
		all_ = len(group)
		overall_ok = 0
		overall_detect = 0
		overall_match = 0
		if not group:
			continue
		hand = group[0]
		img1 = cv2.imread(dir_ + hand, 0)
		for scene in group:
			ok, detect, match = apply_function(dir_ + hand, dir_ + scene) 
			overall_ok = ok and overall_ok + 1 or overall_ok
			overall_detect += detect
			overall_match += match
		print "Group:", names[i], "All:", all_, "OK:", overall_ok, "Detect time:", overall_detect/all_, "Match time:", overall_match/all_

if __name__ == "__main__":
	#harris_corner_measure()
	#sift_surf_measure(show_results=True, operation=cv2.SIFT)
	orb_measure(show_results=True)
	#measure_test_set('orb')
