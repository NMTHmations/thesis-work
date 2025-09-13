from collections import defaultdict, deque
import torch
import supervision as sv
import cv2
import numpy as np
import albumentations as abm
from inference import get_model

class DetermineStrike:
    def __init__(self,start,end,path = None,device = None,debug:bool = False,
                 acceptStart:tuple = None,acceptEnd:tuple = None,
                 lowerHSV:list=None, upperHSV:list = None):
        self.modelPath = "experiment-with-video-frames-po1wn/1"
        self.model = None
        if path == None or device == None:
            self.model = get_model(self.modelPath, api_key="PlEVRUdW9e6KwDkUHIX6")
        else:
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=path)
            self.model.to(device)
        self.tracker = sv.ByteTrack() # create tracker
        self.label_annotator = sv.LabelAnnotator() # get annotations
        self.box_annotator = sv.BoxAnnotator() # get boxing
        self.trace_annotator = sv.TraceAnnotator() # create trace annotator
        self.xList = [i for i in range(start,end)]
        self.positionY = []
        self.PositionX = []
        self.isDebug = False
        if debug == True and lowerHSV != None and upperHSV != None:
            self.isDebug = True
        self.acceptStart = acceptStart
        self.acceptEnd = acceptEnd
        self.lowerHSV = lowerHSV
        self.upperHSV = upperHSV
        
    def _getIntersection(self,p1,p2,p3,p4):
        p1, p2, p3, p4 = map(np.array,(p1,p2,p3,p4))
        d1 = p2 - p1
        d2 = p4 - p3
        denom = (d1[0] * d2[1]) - (d1[1] * d2[0])
        if denom == 0:
            return False, None
        t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / denom
        u = ((p3[0] - p1[0]) * d1[1] - (p3[1] - p1[1]) * d1[0]) / denom
        if u >= 0 and u <= 1 and t >= 0 and t <= 1:
            points = p1 + t * d1
            return True, tuple(points.astype(int))
        return False, None

    def _getMask(self, lower:list, upper:list,iterations:int,frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Define green color range (tune these values)
        lower_yellow = np.array(lower)
        upper_yellow = np.array(upper)
        # Threshold the HSV image to get only green colors
        mask = cv2.inRange(hsv, lower_yellow , upper_yellow)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = cv2.erode(mask, None, iterations=iterations)
        mask = cv2.dilate(mask, None, iterations=iterations)
        _, thresh = cv2.threshold(mask, 85, 255, cv2.THRESH_BINARY)
        points, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE)
        points = [[row[0].tolist()[0][1],row[0].tolist()[0][0]] for row in points]
        return points, thresh
    
    def _getBottom(self,points):
        if len(points) == 0:
            return []
        minimum = points[0]
        for point in points:
            if point[0] < minimum[0]:
                minimum = point
        return [minimum]
    
    def _albumentImage(self,frame):
        blur_image = abm.OneOf([
            abm.MotionBlur(p=0.6),
            abm.GaussianBlur(p=0.6)
            ], p=0.4)
        transformed = blur_image(image=frame)
        frame = transformed["image"]
        return frame
    
    def _getDebugListPoint(self,lista,frame):
        points, thresh = self._getMask(self.lowerHSV,self.upperHSV,1,frame=frame)
        cv2.imshow("threshold",thresh)
        lista = lista + self._getBottom(points)
        return lista
    
    def detectFrame(self,frame:tuple, albumentation:bool = False):
        is_goal = False
        if albumentation:
            frame = self._albumentImage(frame)
        results = self.model.infer(frame, confidence=0.01)[0] # run inference
        detections = sv.Detections.from_inference(results) # get detections
        detections = self.tracker.update_with_detections(detections) # update tracker
        points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        lista = points.tolist()
        if self.isDebug:
            lista = self._getDebugListPoint(lista,frame)
        if (len(lista) != 0):
            print(lista)
            X, Y = lista[0]
            self.PositionX.append(X)
            self.positionY.append(Y)
            coeff = np.polyfit(self.PositionX, self.positionY, 2) # polynomial regression (2nd grade)
            p = np.poly1d(coeff)
            for i in range(1,len(self.PositionX)):
                x = self.PositionX[i]
                x1 = self.PositionX[i-1]
                y = int(p(x))
                y1 = int(p(x1))
                cv2.line(frame,(int(x),int(y)),(int(self.PositionX[i-1]),int(y1)),(255,0,255),2)
            for i in range(1,len(self.xList)+1):
                x = int(self.xList[i-1])
                y = int(p(x))
                if i < len(self.xList):
                    x1 = int(self.xList[i])
                    y1 = int(p(x1))
                    cross, points = self._getIntersection((int(x),int(y)),(x1,y1),self.acceptStart,self.acceptEnd)
                    if cross:
                        is_goal = True
                cv2.circle(frame,(int(x),int(y)),5,(255,0,255),cv2.FILLED)
        cv2.line(frame,self.acceptStart,self.acceptEnd,(0,255,0),2)
        frame = self.box_annotator.annotate(scene=frame, detections=detections)
        frame = self.trace_annotator.annotate(scene=frame, detections=detections)
        return frame, is_goal
    
    def flushPositions(self):
        self.PositionX = []
        self.positionY = []