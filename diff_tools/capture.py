 # -*- coding: utf-8 -*-
"""
Potrafi uchwycić obraz i zapisać do pliku.
Obsługa klawiszem 'q'
Klawisz Esc - wyjście z programu.
"""

import cv2
import sys
import os

shot = False
save_both = False
c = cv2.VideoCapture(0)
if cv2.__version__.find('2.4.8') > -1:
    # reading empty frame may be necessary 
    _, f = c.read()
save_roi = False

PHOTO_DIR = "\\test_images\\"

counter = 0
prefix = "rotated_"
while(1):
    _,f = c.read()
    if save_roi:
        save_image = roi
        name = prefix + "hand%s.jpg" % counter
    else:
        save_image = f
        name = prefix + "scene%s.jpg" % counter

    roi = f[f.shape[0]/6:4*f.shape[0]/6, 
            f.shape[1]/8:4*f.shape[1]/8]
    if shot == True:
        cv2.imwrite(os.path.dirname(__file__) + PHOTO_DIR + name, save_image)
        counter += 1
        shot = False
    if save_both == True:
        name = prefix + "hand%s.jpg" % counter
        name2 = prefix + "scene%s.jpg" % counter
        cv2.imwrite(os.path.dirname(__file__) + PHOTO_DIR + name, roi)
        cv2.imwrite(os.path.dirname(__file__) + PHOTO_DIR + name2, f)
        save_both = False
        counter += 1
    cv2.imshow('ROI', roi)      
    cv2.imshow('IMG',f)
    k = cv2.waitKey(20)	
    if k == 113: #q 
        shot = True
    if k == 114: #r
        save_roi ^= 0x01
        print "Changed, save_roi:", save_roi
    if k == 115: #s
        save_both = True
    if k == 27:
        break

cv2.destroyAllWindows()
c.release()