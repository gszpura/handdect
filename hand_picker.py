import cv2

class HandPicker(object):

    @staticmethod
    def distinguish(img, boxes):
        dy = 10
        dx = 10
        if len(boxes) == 0:
            return None
        if len(boxes) == 1:
            return boxes[0]
        if len(boxes) == 2:
            x1,y1,x2,y2 = boxes[0]
            yp = int(1/float(2)*(y2-y1))
            xp = int(1/float(2)*(x2-x1))
            roi1 = img[y1:y2, x1:x2]
            gray1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            dummy, gray1 = cv2.threshold(gray1, 100, 255, cv2.THRESH_BINARY)
            #contours
            contours, hier = cv2.findContours(gray1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [cnt for cnt in contours if 50 < cv2.contourArea(cnt)]
            print len(cnts), "hand"
            l1 = len(cnts)
            color1 = cv2.cvtColor(gray1, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(color1,cnts,-1,(0,255,0),1)
            cv2.imshow('hand', color1)
            #ratio
            gray1_piece = gray1[yp:yp+dy,xp:xp+dy]
            gray1_all = (x2-x1)*dy
            gray1_white = cv2.countNonZero(gray1_piece)
            ratio1 = gray1_white/float(gray1_all)
            
            x1,y1,x2,y2 = boxes[1]
            yp = int(1/float(8)*(y2-y1))
            xp = int(1/float(2)*(x2-x1))
            roi2 = img[y1:y2, x1:x2]
            gray2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            dummy, gray2 = cv2.threshold(gray2, 100, 255, cv2.THRESH_BINARY)
            #contours
            contours, hier = cv2.findContours(gray2, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = [cnt for cnt in contours if 50 < cv2.contourArea(cnt)]
            print len(cnts), "face"
            l2 = len(cnts)
            color = cv2.cvtColor(gray2, cv2.COLOR_GRAY2BGR)
            cv2.drawContours(color,contours,-1,(0,255,0),1)
            #ratio
            gray2_piece = gray2[yp:yp+dy,xp:xp+dy]
            cv2.imshow('face', color)
            gray2_all = (x2-x1)*dy
            gray2_white = cv2.countNonZero(gray2_piece)
            ratio2 = gray2_white/float(gray2_all)
            
            if l1 < l2:
                return boxes[0]
            return boxes[1]