import cv2
import numpy as np
import time

PREFIX_MODULE = "cascades/"

def w_h_divided_by(image, divisor):
    """Return an image's dimensions, divided by a value."""
    h, w = image.shape[:2]
    return (w/divisor, h/divisor)


class Face(object):
    """Represents face on the image"""
    
    def __init__(self):
        self.faceRect = None
        self.noseRect = None
        self.mouthRect = None
        
    def draw(self, image):
        if self.faceRect is not None:
            x, y, w, h = self.faceRect
            cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0))
            if self.noseRect:
                x, y, w, h = self.noseRect
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0))
            if self.mouthRect:
                x, y, w, h = self.mouthRect
                cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0))
                
                
class Hand(object):
    """Represents hand on the image"""
    
    def __init__(self):
        self.handRect = None
        
    def draw(self, image):
        if self.handRect is not None:
            x, y, w, h = self.handRect
            cv2.rectangle(image, (x, y), (x+w, y+h), (255,0,0))
        
        
        
        
class FaceTracker(object):
    """A tracker for facial features: face, nose, (mouth?)."""
    def __init__(self, scaleFactor = 1.2, minNeighbors = 2, flags = cv2.cv.CV_HAAR_SCALE_IMAGE):
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags
        self._faces = []
        self._faceClassifier = cv2.CascadeClassifier(PREFIX_MODULE + 'haarcascade_frontalface_alt.xml')
        self._eyeClassifier = cv2.CascadeClassifier(PREFIX_MODULE + 'haarcascade_eye.xml')
        self._noseClassifier = cv2.CascadeClassifier(PREFIX_MODULE + 'haarcascade_mcs_nose.xml')
        self._mouthClassifier = cv2.CascadeClassifier(PREFIX_MODULE + 'haarcascade_mcs_mouth.xml')
        
    @property
    def faces(self):
        """The tracked facial features."""
        return self._faces
        
    def update(self, img):
        """Update the tracked facial features."""
        self._faces = []
        cp = img.copy()
        image = cv2.cvtColor(cp, cv2.COLOR_BGR2GRAY)
        image = cv2.equalizeHist(image)

        minSize = w_h_divided_by(image, 6)
        maxSize = w_h_divided_by(image, 2)
        
        cv2.imshow('Equ', image)
        faceRects = self._faceClassifier.detectMultiScale(
            image, self.scaleFactor, self.minNeighbors, self.flags,
            minSize, maxSize)
        print faceRects
        if faceRects is not None:
            for faceRect in faceRects:
                face = Face()
                face.faceRect = faceRect
                x, y, w, h = faceRect
                # Seek a nose in the middle part of the face.
                searchRect = (x+w/4, y+h/4, w/2, h/2)
                """
                face.noseRect = self._detectOneObject(
                    self._noseClassifier, image, searchRect, 32)
                # Seek a mouth in the lower-middle part of the face.
                searchRect = (x+w/6, y+h*2/3, w*2/3, h/3)
                face.mouthRect = self._detectOneObject(
                    self._mouthClassifier, image, searchRect, 16)
                """
                self._faces.append(face)
                
                
    def _detectOneObject(self, classifier, image, rect, ratio):
        x, y, w, h = rect
        minSize = w_h_divided_by(image, ratio)
        subImage = image[y:y+h, x:x+w]
        subRects = classifier.detectMultiScale(
            subImage, self.scaleFactor, self.minNeighbors,
            self.flags, minSize)
        if len(subRects) == 0:
            return None
        subX, subY, subW, subH = subRects[0]
        return (x+subX, y+subY, subW, subH)

      
class HandTracker(object):
    """A tracker for hand features"""
    def __init__(self, scaleFactor = 1.1, minNeighbors = 4, flags = cv2.cv.CV_HAAR_DO_CANNY_PRUNING):
        #name = PREFIX_MODULE + "haarcascade_hand_2.xml"
        #name = PREFIX_MODULE + "fist.xml"
        #name = PREFIX_MODULE + "palm.xml"
        name = PREFIX_MODULE + "hand_cascade.xml"
        
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors
        self.flags = flags
        self._hands = []
        self._handClassifier = cv2.CascadeClassifier(name)
        
    @property
    def hands(self):
        """The tracked facial features."""
        return self._hands
        
        
    def update(self, img):
        """Update the tracked facial features."""
        self._hands = []
        cp = img.copy()
        image = cv2.cvtColor(cp, cv2.COLOR_BGR2GRAY)                
        handRects = self._handClassifier.detectMultiScale(
            image, self.scaleFactor, self.minNeighbors, self.flags,
            (70, 70), (175, 125))
        for r in handRects:
            h = Hand()
            h.handRect = r
            self._hands.append(h)
    