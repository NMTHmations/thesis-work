from collections import defaultdict, deque
import torch
import supervision as sv
import cv2
import numpy as np
import albumentations as abm
from inference import get_model

class DetermineStrike:
    def __init__(self,start,end,height:int, width:int,debug:bool = False,
                 acceptStart:tuple = None,acceptEnd:tuple = None,
                 lowerHSV:list=None, upperHSV:list = None):
        self.tracker = sv.ByteTrack(track_activation_threshold=0.25,minimum_matching_threshold=1) # create tracker
        self.label = sv.LabelAnnotator() # get annotations
        self.annotator = sv.BoxAnnotator() # get boxing
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
        self.height = height
        self.width = width
        self.class_names = ['Ball','Football']
        
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
    
    def extract_detections(self,output:list, h: int, w: int, threshold: float = 0.05):
        xyxy = []
        confidence = []
        class_id = []
        num_detections = 0

        for i, detections in enumerate(output):
            if len(detections) == 0:
                continue
            for detection in detections:
                if len(detection) == 0:
                    continue
                bbox, score = detection[:4], detection[4]

                if score < np.float32(threshold):
                    continue

                bbox[0], bbox[1], bbox[2], bbox[3] = (
                    bbox[1] * w,
                    bbox[0] * h,
                    bbox[3] * w,
                    bbox[2] * h,
                ) 

                xyxy.append(bbox)
                confidence.append(score)
                class_id.append(i)
                num_detections += 1
        return {
            "xyxy": np.array(xyxy),
            "confidence": np.array(confidence, dtype=np.float32),
            "class_id": np.array(class_id),
            "num_detections": num_detections
        }

    def process_detections(self,frame: np.ndarray, detections: dict, class_names: list, tracker: sv.ByteTrack, 
                           box_annotator: sv.BoxAnnotator, label_annotator: sv.LabelAnnotator):
        if len(detections['xyxy']) == 0:
            return frame, np.empty((0, 2))
        sv_detections = sv.Detections(xyxy=detections["xyxy"],
                                      confidence=detections["confidence"],
                                      class_id=detections["class_id"])

        sv_detections = tracker.update_with_detections(sv_detections)

        points = sv_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        labels = [f"{class_names[cls]} {conf:.2f}" for cls, conf in zip(sv_detections.class_id, sv_detections.tracker_id)]

        annotated_frame = box_annotator.annotate(scene=frame.copy(),detections=sv_detections)

        annotated_frame = label_annotator.annotate(scene=annotated_frame,detections=sv_detections,labels=labels)
        return (annotated_frame, points)
    
    def detectFrame(self,frame:tuple, detections, albumentation:bool = False):
        is_goal = False
        if albumentation:
            frame = self._albumentImage(frame)
        sv_detections = self.extract_detections(detections,self.height,self.width,0.25)
        frame, points = self.process_detections(frame,sv_detections,self.class_names,self.tracker,self.annotator,self.label)
        """
        lista = points.tolist()
        if self.isDebug:
            lista = self._getDebugListPoint(lista,frame)
        if (len(lista) != 0):
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
        cv2.line(frame,self.acceptStart,self.acceptEnd,(0,255,0),2)"""
        return frame, is_goal
    
    def flushPositions(self):
        self.PositionX = []
        self.positionY = []