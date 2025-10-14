from abc import abstractmethod
from typing import Optional, Tuple

import cv2
import numpy as np
import supervision as sv

from project.detection.types.enums import ModelTypes, FrameSize


class DetectionModel:
    @abstractmethod
    def infer(self, frame):
        pass

    @abstractmethod
    def batch_infer(self, batch):
        pass

    @abstractmethod
    def getDetectionFromResult(self, result, originalWH : Optional[Tuple[int,int]]) -> sv.Detections:
        pass

    def _remapDetectionsIfNeeded(self, det : sv.Detections, originalWH : Tuple[int,int]):
        try:
            xy = np.array(det.xyxy)
            #640 + 1, because frames got resized at 640x640
            if xy.size and xy.max() <= FrameSize.INPUTDIM + 1:
                det.xyxy = self._mapDetectionsToOriginal(xy, originalWH)
        except Exception:
            return det

        return det

    def _mapDetectionsToOriginal(self, xyxy, originalWH : Tuple[int,int]):
        width, height = originalWH
        inputDimension = float(FrameSize.INPUTDIM)
        scaleX = width/inputDimension
        scaleY = height/inputDimension

        xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * scaleX
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * scaleY
        # clamp
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, width - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, height - 1)
        return xyxy


class DetectionModelFactory:
    @staticmethod
    def create(**kwargs) -> DetectionModel:
        modelType = kwargs["modelType"]

        if modelType == ModelTypes.YOLO:

            modelPath = kwargs["modelPath"]
            device = kwargs["device"]
            return YOLOModel(modelPath=modelPath, device=device, inferenceImgSize=640)

        elif modelType == ModelTypes.INFERENCE:

            modelPath = kwargs["modelPath"]
            apiKey = kwargs["apiKey"]
            return RoboflowModel(modelPath=modelPath, apiKey=apiKey)

        elif modelType == ModelTypes.COLOR:
            loverHSV = kwargs["loverHSV"]
            upperHSV = kwargs["upperHSV"]

            return ColorDetectorModel(loverHSV=loverHSV, upperHSV=upperHSV)
        else:
            raise ValueError("Unknown model type")



class ColorDetectorModel(DetectionModel):
    def __init__(self, loverHSV, upperHSV):
        self.loverHSV = loverHSV
        self.upperHSV = upperHSV

    def infer(self, frame):
        mask = self._getMask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


        return contours

    def batch_infer(self, batch):
        contoursBatch = []
        for frame in batch:
            contoursBatch.append(self.infer(frame))

        return contoursBatch


    def getDetectionFromResult(self, result, originalWH: Optional[Tuple[int, int]]) -> sv.Detections:
        bboxList = []
        confList = []
        classIDList = []

        detections = sv.Detections.empty()

        for r in result:
            if cv2.contourArea(r) > 100:
                x, y, w, h = cv2.boundingRect(r)
                bboxList.append([x, y, x + w, y + h])  # xyxy formátum
                classIDList.append(0)  # ha van osztály, ide teheted a class ID-t
                confList.append(1)


        if len(bboxList) > 0:
            bboxes = np.array(bboxList, dtype=int)
            confidences = np.array(confList, dtype=int)
            classIDList = np.array(classIDList, dtype=int)

            detections = sv.Detections(xyxy=bboxes, confidence=confidences, class_id=classIDList)

        return detections

    def _getMask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, np.array(self.loverHSV), np.array(self.upperHSV))

        #Remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask


class YOLOModel(DetectionModel):
    def __init__(self, modelPath : str, device : Optional[str | int], inferenceImgSize : Optional[int]):
        from ultralytics import YOLO

        self.model = YOLO(modelPath)

        if device is None:
            self.device = "cpu"
        else:
            self.device = device
        if inferenceImgSize is None:
            self.inferenceImgSize = FrameSize.INPUTDIM
        else:
            self.inferenceImgSize = inferenceImgSize

        self.isVerbose = False

    def infer(self, frame):
        return self.model.predict(frame, imgsz=self.inferenceImgSize, verbose=self.isVerbose, device=self.device)

    def batch_infer(self, batch):
        return self.model.predict(batch, imgsz=self.inferenceImgSize, verbose=self.isVerbose, device=self.device)

    def getDetectionFromResult(self, result, originalWH : Optional[Tuple[int,int]] = None) -> sv.Detections:
        if originalWH is None:
            originalWH = (FrameSize.INPUTDIM,FrameSize.INPUTDIM)

        detection = sv.Detections.from_ultralytics(result)

        return self._remapDetectionsIfNeeded(detection, originalWH)


class RoboflowModel(DetectionModel):
    def __init__(self, modelPath : str, apiKey : str):
        from inference import get_model, Model

        self.model : Model = get_model(modelPath, api_key=apiKey)

    def infer(self, frame):
        return self.model.infer(frame)

    def batch_infer(self, batch):
        return self.model.infer(batch)

    def getDetectionFromResult(self, result, originalWH : Optional[Tuple[int,int]]) -> sv.Detections:
        if originalWH is None:
            originalWH = (FrameSize.INPUTDIM,FrameSize.INPUTDIM)

        det = sv.Detections.from_inference(result)

        return self._remapDetectionsIfNeeded(det, originalWH)
