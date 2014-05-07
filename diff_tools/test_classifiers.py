import cv2
import numpy as np
from pdym.main_utils import find_contours, \
	get_biggest_cnt, \
	split_into_planes, \
	find_head_with_otsu, \
	calculate_histogram, \
	get_roi, \
	init_camera, \
	release_camera, \
	save_image, \
	read_image, \
	get_H_channel
from histograms import hist1d
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

camera = init_camera()


def get_frame():
    _, frame = camera.read()
    return frame


def produce_histogram_from_image(image, channel="H"):
	H, S, V, y, u, v = split_into_planes(image)
	planes_map = {"H":H, "S":S, "V":V, "Y":y, "U":u, "v": v}
	head_as_mask, rect = find_head_with_otsu(V)
	roi = get_roi(planes_map[channel], rect)
	cv2.imshow('head', roi)
	cv2.waitKey(0)
	histogram = calculate_histogram(roi, head_as_mask)
	return histogram, head_as_mask, rect


def non_head_mask_from_head_rect(image, head_rect):
	non_head_mask = np.zeros(image.shape, np.uint8)
	x, y, w, h = head_rect
	non_head_mask[y-10:y+h+100, x:x+w] = 255
	return cv2.bitwise_not(non_head_mask)


def count_misclassification_ratio(classified, head, head_rect):
	x, y, w, h = head_rect
	head_classified = classified[y:y+h, x:x+w].copy()
	head_classified = cv2.bitwise_and(head_classified, head)
	classified[y:y+h, x:x+w] = 0
	miss_count = cv2.countNonZero(classified)
	head_count = cv2.countNonZero(head)
	correct_count = cv2.countNonZero(head_classified)
	print "Missclassified:", miss_count, "Correct:", correct_count, "of:", head_count


def classify_linear(image, ranges):
	"""
	@param image: single channel image
	@param ranges: ranges in the form of list: [(0, 30), (110, 140)]
	"""
	classified = np.zeros(image.shape, np.uint8)
	for x, y in ranges:
		part = cv2.inRange(image, np.array(x, np.uint8), 
                                  np.array(y, np.uint8))
		classified = cv2.bitwise_or(part, classified)
	cv2.imshow('Classified image - Linear', classified)
	cv2.waitKey(0)
	return classified

def classify_bayes(image, hist_cmp):
	"""
	@param image: single channel image
	@param hist_cmp: P(x | skin)/P(x | non-skin) > tau table
	"""
	classified = hist_cmp[image]
	cv2.imshow('Classified image - Bayes', classified)
	cv2.waitKey(0)
	return classified


def linear_simple_classifier(frame, channel="H", cut_off=0.05, weight_factor=0.8, selectivity=5):
	# init
	hist, head, head_rect = produce_histogram_from_image(frame, channel)
	H, S, V, y, u, v = split_into_planes(frame)
	planes_map = {"H": H, "S": S, "V": V, "y":y, "u": u, "v": v }
	hist = list(zip(*hist)[0])

	# cut off small values
	cut_off_value = cut_off*max(hist)
	hist = [(i, v) for i, v in enumerate(hist) if v > cut_off_value]

	# select ranges
	ranges = []
	current = -1
	total_weight = sum(zip(*hist)[1])
	current_weight = 0
	for i, v in hist:
		current_weight += v
		if i - current < selectivity or current < 0:
			current = i
		else:
			ranges.append(current)
			break
		if current_weight > weight_factor*total_weight:
			ranges.append(current)
			break
		if len(ranges) == 0:
			ranges.append(current)

	# classify and stats
	print ranges
	i = iter(ranges)
	cls = classify_linear(planes_map[channel], zip(i, i))
	count_misclassification_ratio(cls, head, head_rect)


def linear_extended_classifier(frame, channel="H", cut_off=0.05, weight_factor=1.0, selectivity=3):
	# init
	hist, head, head_rect = produce_histogram_from_image(frame, channel)
	H, S, V, y, u, v = split_into_planes(frame)
	planes_map = {"H": H, "S": S, "V": V, "y":y, "u": u, "v": v }
	hist = list(zip(*hist)[0])

	# cut off small values
	cut_off_value = cut_off*max(hist)
	hist = [(i, v) for i, v in enumerate(hist) if v > cut_off_value]
	print hist
	# select ranges
	total_weight = sum(zip(*hist)[1])
	ranges = []
	current = -1
	current_weight = 0
	for i, v in hist:
		current_weight += v
		if i - current <= selectivity or current < 0:
			current = i
		else:
			ranges.append(current)
			if len(ranges) >= 20:
				break
			ranges.append(i)
			current = i
		if current_weight > weight_factor*total_weight:
			ranges.append(current)
			break
		if len(ranges) == 0:
			ranges.append(current)
	if len(ranges) % 2 != 0:
		ranges.append(ranges[-1])
	
	# select two ranges with highest weights
	def range_weight(rng):
		return sum(v for i, v in hist if rng[1] >= i >= rng[0])
	i = iter(ranges)
	print ranges
	rngs = zip(i, i)
	rngs = sorted([(range_weight(rng), rng) for rng in rngs], reverse=True)[:2]
	rngs = zip(*rngs)[1]

	# classify and statss
	print rngs
	cls = classify_linear(planes_map[channel], rngs)
	count_misclassification_ratio(cls, head, head_rect)


def bayes_classifier(frame, channel="H", tau=1.51):
	# init
	hist, head, head_rect = produce_histogram_from_image(frame, channel)
	H, S, V, y, u, v = split_into_planes(frame)
	planes_map = {"H": H, "S": S, "V": V, "y":y, "u": u, "v": v }

	# histograms
	hist = np.array(list(zip(*hist)[0]))
	non_head_mask = non_head_mask_from_head_rect(planes_map[channel], head_rect)
	non_head_hist = calculate_histogram(planes_map[channel], non_head_mask)
	non_head_hist = np.array(list(zip(*non_head_hist)[0]))

	# normalization
	norm_non_head_hist = non_head_hist/non_head_hist.sum()
	norm_hist = hist/hist.sum()
	norm_hist[norm_hist == 0] = 0.00001
	norm_non_head_hist[norm_non_head_hist == 0] = 0.00001
	
	# Bayes equasion
	hist_cmp = norm_hist/norm_non_head_hist

	hist_cmp[hist_cmp > tau] = 255
	hist_cmp[hist_cmp != 255.0] = 0
	hist_cmp = hist_cmp.astype(np.uint8)

	# classify and stats
	cls = classify_bayes(planes_map[channel], hist_cmp)
	count_misclassification_ratio(cls, head, head_rect)


if __name__ == "__main__":
	frame = get_frame()
	#frame = read_image()
	channel = "H"
	linear_simple_classifier(frame.copy(), channel)
	linear_extended_classifier(frame.copy(), channel)
	bayes_classifier(frame.copy(), channel)
	#classify_linear(get_H_channel(frame), [(1, 100)])
	release_camera(camera)