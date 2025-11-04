import supervision as sv
import cv2
import numpy as np
import albumentations as abm
from project.detection.final.predictor import KFPredictor
from project.detection.final.finalSingleCam import getCenter

class DetermineStrike:
    def __init__(self,
                 acceptStart:tuple = None,acceptEnd:tuple = None, isFront:bool = False, dt=1.0 / 60.0, debug = False):
        self.debug = debug
        self.tracker = sv.ByteTrack(track_activation_threshold=0.25,minimum_matching_threshold=1) if self.debug == False else sv.ByteTrack()
        self.label = sv.LabelAnnotator()
        self.annotator = sv.BoxAnnotator()
        self.trace_annotator = sv.TraceAnnotator()
        self.xList = [i for i in range(0,640)]
        self.positionY = []
        self.PositionX = []
        self.isDebug = False
        self.acceptStart = acceptStart
        self.acceptEnd = acceptEnd
        self.isFront = isFront
        self.PolinomialDegree = self._setDegree()
        self.height = 480
        self.width = 640
        self.center = ()
        self.prevCenter = ()
        self.predictor = KFPredictor(dt=dt)
        self.class_names = ['Ball','Football']
        self.still_counter = 0
    
    def _setDegree(self):
        if self.isFront == False:
            return 2
        return 1
    
    def filterPredictionsKarman(self):

        goalLine = (self.acceptStart, self.acceptEnd)

        impactPoint, mappedImpact = None, None

        if self.center != () and self.prevCenter != () and (
                abs(self.center[0] - self.prevCenter[0]) > 5.0 or abs(self.center[1] - self.prevCenter[1]) > 5.0):
            impactPoint, mappedImpact = self.predictor.predictImpact(self.center, goalLine)
            if impactPoint is not None:
                lastImpactPoint = tuple(map(int, impactPoint))
                lastMappedImpactPoint = float(mappedImpact)
                return lastImpactPoint, lastMappedImpactPoint
        return None, None

    
    def _getBounce(self, window:int = 5, threshold:float = 2.0):
        if self.isFront == False:
            if len(self.positionY) < window + 2:
                return False
        else:
            if len(self.PositionX) < window + 2:
                return False
        
        recent = None
        if self.isFront:
            recent = self.PositionX[-(window+2):]
        else:
            recent = self.positionY[-(window+2):]
        
        dy = np.diff(recent)
        for i in range(1, len(dy)):
            if dy[i-1] > threshold and dy[i] < -threshold:
                return True

        return False

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
    
    def _albumentImage(self,frame):
        blur_image = abm.OneOf([
            abm.MotionBlur(p=0.6),
            abm.GaussianBlur(p=0.6)
            ], p=0.4)
        transformed = blur_image(image=frame)
        frame = transformed["image"]
        return frame

    
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
        sv_detections = detections
        if self.debug == False:
            if len(detections['xyxy']) == 0:
                return frame, np.empty((0, 2))
            sv_detections = sv.Detections(xyxy=detections["xyxy"],
                                          confidence=detections["confidence"],
                                          class_id=detections["class_id"])

        sv_detections = tracker.update_with_detections(sv_detections)

        points = sv_detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

        annotated_frame = frame

        annotated_frame = box_annotator.annotate(scene=frame.copy(),detections=sv_detections)

        if self.debug == False:

            labels = [f"{class_names[cls]} {conf:.2f}" for cls, conf in zip(sv_detections.class_id, sv_detections.tracker_id)]

            annotated_frame = label_annotator.annotate(scene=annotated_frame,detections=sv_detections,labels=labels)
        return (annotated_frame, points)
    
    def checkStill(self):
        motion_inactive = False
        if len(self.PositionX) > 2:
            vx = abs(self.PositionX[-1] - self.PositionX[-2])
            vy = abs(self.positionY[-1] - self.positionY[-2])
            if vx < 3.0 or vy < 3.0:
                self.still_counter += 1
            else:
                self.still_counter = 0
                motion_inactive = False
        if self.still_counter >= 10:
            motion_inactive = True
        return motion_inactive
    
    def detectFrame(self,frame:tuple, detections, albumentation:bool = False):
        cross_points = None
        if albumentation:
            frame = self._albumentImage(frame)
        sv_detections = self.extract_detections(detections,self.height,self.width,0.25) if self.debug == False else detections
        frame, points = self.process_detections(frame,sv_detections,self.class_names,self.tracker,self.annotator,self.label)
        lista = points.tolist()
        if self.debug:
            self.prevCenter = self.center
            self.center = ()
            try:
                self.center = getCenter(sv_detections.xyxy[0])
                lastImpactPoint, lastMappedImpactPoint = self.filterPredictionsKarman()
                cross_points = lastImpactPoint
            except:
                pass
        else:
            if (len(lista) != 0):
                X, Y = lista[0]
                if self.checkStill():
                    return frame, None
                self.PositionX.append(X)
                self.positionY.append(Y)
                if self._getBounce():
                    self.PolinomialDegree = min(self.PolinomialDegree + 1, 6)
                coeff = np.polyfit(self.PositionX, self.positionY, self.PolinomialDegree)
                p = np.poly1d(coeff)
                try:
                    for i in range(1,len(self.xList)+1):
                        x = int(self.xList[i-1])
                        y = int(p(x))
                        if i < len(self.xList):
                            x1 = int(self.xList[i])
                            y1 = int(p(x1))
                            cross, points = self._getIntersection((x,y),(x1,y1),self.acceptStart,self.acceptEnd)
                            if cross:
                                cross_points = points
                        cv2.circle(frame,(int(x),int(y)),5,(255,0,255),cv2.FILLED)
                except:
                    pass
            else:
                if len(self.PositionX) > 10:
                    self.flushPositions()
        cv2.line(frame,self.acceptStart,self.acceptEnd,(0,255,0),2)
        return frame, cross_points
    
    def flushPositions(self):
        self.PositionX = []
        self.positionY = []
        self.PolinomialDegree = self._setDegree()
        self.center = ()
        self.prevCenter = ()
        self.still_counter = 0
