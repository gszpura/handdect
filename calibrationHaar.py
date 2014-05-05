import numpy as np
import cv2
import cv2.cv as cv

from calibration2 import Calibration
from main_utils import draw_boxes, get_biggest_cnt
from config import HEIGHT, WIDTH

def detect(img, cascade):
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30), flags = cv.CV_HAAR_SCALE_IMAGE)
    if len(rects) == 0:
        return []
    rects[:,2:] += rects[:,:2]
    return rects

def draw_mask(img, rect):
	x, y, w, h = rect
	img[y:y+h, x:x+w] = (255, 255, 255)
	return img

class CalibrationHaar(Calibration):


	def __init__(self):
		self.head = "cascades/haarcascade_frontalface_alt.xml"
		self.head_cascade = cv2.CascadeClassifier(self.head)
		
		super(CalibrationHaar, self).__init__()
		self.planes = "image"

	def get_non_head_mask(self, box):
		"""
		Finds mask for non head pixels
		"""
		mask = np.zeros((self.h, self.w), np.uint8)
		x1, y1, x2, y2 = box
		mask[y1:y2, x1:x2] = 255
		#cv2.imshow('non head mask', cv2.bitwise_not(mask))
		return cv2.bitwise_not(mask)

	def discover_regions(self, img):
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)
		rects = detect(gray, self.head_cascade)
		if rects != []:
			draw_boxes(img, rects)
			x1, y1, x2, y2 = rects[0]
			h = y2 - y1
			w = x2 - x1
			self.rect = [x1, y1, w, h]

			mask = np.zeros((h, w), np.uint8)
			mask[:, int(0.2*w):w-int(0.2*w)] = 255
			non_head_mask = self.get_non_head_mask([x1, y1 - int(0.1*h), 
													x2, y2 + int(0.1*h)])
			#img2 = gray.copy()
			#img2[:,:] = 0
			#img2[y1:y2, x1:x2] = mask
			#cv2.imshow('head mask', img2)
			return mask, non_head_mask
		return None, None
        


def test_main():
	c = cv2.VideoCapture(0)
	if cv2.__version__.startswith('2.4.8'):
		_, f = c.read()

	clbr = CalibrationHaar()
	while(1):
		_, f = c.read()
		clbr.discover_regions(f)
		cv2.imshow('HEAD', f)
		k = cv2.waitKey(20)
		if k == 27:
			break   
	cv2.destroyAllWindows()
	c.release()

#test_main()