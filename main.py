"""
Methods:
* Background substraction + HSV
* HAAR cascades

Last updated: 03-05-2014
"""
import cv2
import os
import sys
import time

from tracker import TrackerAL, StateTracker
from transformer import Transformer
from haartrack import FaceTracker, Face, Hand, HandTracker
from hand_picker import HandPicker
from main_utils import draw_boxes
from calibration2 import Calibration
from calibrationHaar import CalibrationHaar


#global init
shot = False
c = cv2.VideoCapture(0)
if cv2.__version__.startswith('2.4.8'):
    _,f = c.read()

#version for Cascades
def mainCascades():
    track = TrackerAL()
    hnd = HandTracker()

    while(1):    
        _,f = c.read()
        st = time.time()
        small = cv2.resize(f, (320, 240))
        hnd.update(small)
        print time.time() - st
        hands = hnd.hands
        boxes = track.pbb(hands)
        box = HandPicker.distinguish(small, boxes)
        if box:
            draw_boxes(small, [box])
        cv2.imshow('IMG', small)
        k = cv2.waitKey(20)	
        if k == 113: #q 
            shot = True
        if k == 27:
            break   
    cv2.destroyAllWindows()
    c.release()


def read_image():
    #TODO: implement me
    return None


def read_camera():
    _, f = c.read()
    return f


def run_calibration():
    """
    Runs calibration.
    """
    clbr = Calibration()
    cnt = 0
    while (not clbr.end):
        img = read_camera()
        if img is None:
            cnt += 1
        clbr.update(img)
        if cnt > 100:
            return None
    return clbr


#version with Substraction and HSV detection
def mainSubHSV(profile=0):
    clbr = run_calibration()

    print "*******", clbr.conf_h, clbr.conf_yv, clbr.thr, clbr.light, "*******"
    trf = Transformer(clbr.light, clbr.conf_h, clbr.conf_yv, clbr.thr)
    trf.turn_on_bayes_classifier(clbr.pdf_cmp_h, clbr.pdf_cmp_v)
    track = StateTracker()

    while (1):
        f = read_camera()
        
        move_cue = trf.move_cue(f)
        #t1 = time.time()
        #skin_cue = trf.bayes_skin_classifier(f)
        skin_cue = trf.linear_skin_classifier(f)
        #print time.time() - t1
        final = cv2.bitwise_and(skin_cue, move_cue)
        track.update(final)
        info = track.follow(f)
        
        cv2.imshow('IMG', f)
        cv2.imshow('SKIN FINAL', final)
        k = cv2.waitKey(20)
        if k == 113: #q
            shot = True
        if k == 27:
            break
        # debug & profile part
        if profile > 0:
            profile -= 1
            if profile == 0:
                break

    cv2.destroyAllWindows()
    c.release()

    
    
if __name__ == "__main__":
    mainSubHSV()