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

from tracker import TrackerAL, TrackerNext, StateTracker
from transformer import Transformer
from haartrack import FaceTracker, Face, Hand, HandTracker
from hand_picker import HandPicker
from main_utils import draw_boxes
from calibration import Calibration
from calibration2 import Calibration2


#global init
shot = False
c = cv2.VideoCapture(0)

#version for Cascades
def mainCascades():
    track = TrackerAL()
    trf = Transformer()
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
CFG_HSV = [1, 2, 145, 200]
CFG_THR = 90
LIGHT = "Night"

def mainSubHSV():
    clbr = Calibration2()
    while (not clbr.end):
        _,f = c.read()
        clbr.update(f)
    print clbr.best_conf, clbr.thr, clbr.light, "*******&&&&&&"
    LIGHT = clbr.light
    CFG_HSV = clbr.best_conf
    CFG_THR = clbr.thr
    track = StateTracker(LIGHT, CFG_HSV, CFG_THR)
    trf = Transformer(LIGHT)
    trf.set_color_ranges(CFG_HSV)
    while (1):
        _,f = c.read()
        st = time.time()
        move_cue = trf.move_cue(f)
        skin_cue = trf.skin_color_cue(f)
        final = trf.smart_and(move_cue, skin_cue)
        #final = trf.postprocess(final)
        track.update(final)
        track.follow(f)
        print time.time() - st
        cv2.imshow('IMG', f)
        cv2.imshow('IMG2', final)
        k = cv2.waitKey(20)
        if k == 113: #q
            shot = True
        if k == 27:
            break


    cv2.destroyAllWindows()
    c.release()

    
    
if __name__ == "__main__":
    mainSubHSV()