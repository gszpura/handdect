import timeit
import time
import cv2
#command line:
#python -m timeit '"-".join(str(n) for n in range(100))'

####time module init
PHOTO_PATH = "C:\\Python27\\Pracad\\"
PHOTO = "faceBlur2.png"
img = cv2.imread(PHOTO_PATH + PHOTO)
img2 = cv2.imread(PHOTO_PATH + "face.png")
from haartrack import FaceTracker
#c = cv2.VideoCapture(0)
#c.release()
#####

start = time.time()
####begining
#img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
f = FaceTracker()
####end
stop = time.time()
print "Use of time module:"
print (stop - start)

t = timeit.Timer("res = cv2.medianBlur(img,7)", "import cv2; img = cv2.imread('C:\\Python27\\Pracad\\face.png')")
print '\n\n\n'
print 'TIMEIT:'
print t.timeit(500)

print 'REPEAT:'
print t.repeat(3, 2)