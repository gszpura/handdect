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

# Websockets imports
import json
import threading
import signal, sys, ssl, logging
from SimpleWebSocketServer import WebSocket, SimpleWebSocketServer, SimpleSSLWebSocketServer


PHOTO_DIR = "\\diff_tools\\test_images\\"
PATH = os.path.dirname(__file__) + PHOTO_DIR


#global init
shot = False
c = cv2.VideoCapture(0)
if cv2.__version__.startswith('2.4.8'):
    _,f = c.read()

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

class SimpleEcho(WebSocket):

    def handleMessage(self):        
        if self.data is None:
            self.data = ''        
        try:
            self.sendMessage(str(self.data))
        except Exception as n:
            print n
            
    def handleConnected(self):     
        print self.address, 'connected'

    def handleClose(self):
        print self.address, 'closed'

cls = SimpleEcho


class WebSocketThread(threading.Thread):    

    def __init__(self, threadID, name, counter):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name = name
        self.counter = counter
        self.server = None
        self.host = "192.168.0.100"
        self.port = 8090

    def run(self):
        print "Starting " + self.name
        self.server = SimpleWebSocketServer(self.host, self.port, cls) 
        self.server.serveforever()
        print "Exiting " + self.name
    
    def close_sig_handler(self, signal, frame):
        self.server.close()      
    
    def pushMessage(self, coords, gesture):
        if coords:
            print "Msg:", coords, gesture

        if self.server == None:
            return        

        for client in self.server.connections.itervalues():
            if client != self:
                try: 
                    msg = str(json.dumps({
                                            "params": {
                                                "x":coords[0],
                                                "y":coords[1],
                                                "w":coords[2], 
                                                "h":coords[3],
                                                "gesture":gesture, 
                                                "resize": True,
                                            },
                                            'event':"g0", 
                                            'timestamp': int(time.time()),
                                            'cmd':'trigger'
                                        })
                             )
                    client.sendMessage(msg)
                except Exception as n:
                    print n

    def printConnections(self):
        print self.server.connections


def read_image():
    f = cv2.imread(PATH + "back_scene10.jpg")    
    return f


def read_camera():
    _, f = c.read()
    return f


def run_calibration():
    """
    Runs calibration process.
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

    # WebSockets
    websocket_thread = WebSocketThread(1, "WebSocketThread", 1)
    signal.signal(signal.SIGINT, websocket_thread.close_sig_handler) 
    websocket_thread.start()

    while (1):
        f = read_camera()

        move_cue = trf.move_cue(f)
        #skin_cue = trf.bayes_skin_classifier(f)
        skin_cue = trf.linear_skin_classifier(f)
        final = cv2.bitwise_and(skin_cue, move_cue)
        track.update(final)

        coords, gesture = track.follow(f)
        # send info with WebSockets
        websocket_thread.pushMessage(coords, gesture)   

        cv2.imshow('IMG', f)
        cv2.imshow('SKIN FINAL', final)
        k = cv2.waitKey(20)
        if k == 113: #q
            shot = True
        if k == 27:
            websocket_thread.server.close()
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