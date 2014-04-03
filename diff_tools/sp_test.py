from time import time
import cv2

from body_model_py import Leaf as PL
from body_model import Leaf as CL,  test_run, BodyPartsModel


img = cv2.imread('img13o.png')
img =  cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print "#########test_run begin########"
test_run()
print "#########test_run end##########"
amount = img.shape[0]/2
jump = 2
cum = 0
t1 = time()
for i in range(0, amount):
	sl = img[jump*i:jump*(i+1)][0]
	t1 = time()
	PL(sl)
	cum += time() - t1
print cum/amount
print cum

print "*************"
amount = img.shape[0]/2
jump = 2
cum = 0
t1 = time()
for i in range(0, amount):
	sl = img[jump*i:jump*(i+1)][0]
	t1 = time()
	CL(sl.tolist())
	cum += time() - t1

print cum/amount
print cum

#import cProfile
#cProfile.run('BodyPartsModel(img)')

t1 = time()
a = BodyPartsModel(img)
cum = time() - t1
print cum