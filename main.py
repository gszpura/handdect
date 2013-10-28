"""
Methods:
* Background substraction + HSV
* HAAR cascades

Last updated: 26-10-2013
"""
import cv2
import os
import sys
import time

from tracker import TrackerAL, Tracker, TrackerNext, StateTracker
from transformer import Transformer
from haartrack import FaceTracker, Face, Hand, HandTracker
from hand_picker import HandPicker
from main_utils import draw_boxes


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
def mainSubHSV():
    track = StateTracker()
    trf = Transformer()

    while(1):    
        _,f = c.read()
        st = time.time()
        small = cv2.resize(f, (320, 240))
        move_cue = trf.move_cue(f)
        box = track.get_hsv_limits(move_cue)
        skin_cue = trf.skin_color_cue(f, box)
        final = trf.smart_and(move_cue, skin_cue)
        #final = trf.postprocess(final)
        track.update(final)
        track.distinguish(f)
        #print time.time() - st
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