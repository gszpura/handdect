"""
Methods:
* Background substraction + HSV
* HAAR cascades

Last updated: 10-12-2013
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
from calibration import Calibration
from calibration2 import Calibration2


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


#version with Substraction and HSV detection
def mainSubHSV(profile=0):
    clbr = Calibration2()
    while (not clbr.end):
        _,f = c.read()
        clbr.update(f)
    print "*******", clbr.conf_h, clbr.conf_yv, clbr.thr, clbr.light, "*******"
    LIGHT = clbr.light
    CFG_HSV = clbr.conf_h
    CFG_YUV = clbr.conf_yv
    CFG_THR = clbr.thr
    track = StateTracker()
    trf = Transformer(LIGHT, CFG_HSV, CFG_YUV, CFG_THR)
    while (1):
        _,f = c.read()
        st = time.time()
        move_cue = trf.move_cue(f)
        skin_cue = trf.skin_color_cue(f)
        final = cv2.bitwise_and(skin_cue, move_cue)
        track.update(final)
        info = track.follow(f)
        #print time.time() - st
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