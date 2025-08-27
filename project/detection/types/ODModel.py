from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np
import supervision as sv
from inference.core.workflows.core_steps.common.query_language.operations import detections

from project.detection.types.enums import ModelTypes


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
            if xy.size and xy.max() <= 641:
                det.xyxy = self._mapDetectionsToOriginal(xy, originalWH)
        except Exception:
            return det

        return det

    def _mapDetectionsToOriginal(self, xyxy, originalWH : Tuple[int,int]):
        width, height = originalWH
        scaleX = width/640.0
        scaleY = height/640.0

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
        modelPath = kwargs["modelPath"]
        if modelType == ModelTypes.YOLO:
            device = kwargs["device"]
            return YOLOModel(modelPath=modelPath, device=device, inferenceImgSize=640)
        elif modelType == ModelTypes.INFERENCE:
            apiKey = kwargs["apiKey"]
            return RoboflowModel(modelPath=modelPath, apiKey=apiKey)
        else:
            raise ValueError("Unknown model type")





class YOLOModel(DetectionModel):
    def __init__(self, modelPath : str, device : Optional[str | int], inferenceImgSize : Optional[int]):
        from ultralytics import YOLO

        self.model = YOLO(modelPath)

        if device is None:
            self.device = "cpu"
        else:
            self.device = device

        if inferenceImgSize is None:
            self.inferenceImgSize = 640
        else:
            self.inferenceImgSize = inferenceImgSize

        self.isVerbose = False

    def infer(self, frame):
        return self.model.predict(frame, imgsz=self.inferenceImgSize, verbose=self.isVerbose, device=self.device)

    def batch_infer(self, batch):
        return self.model.predict(batch, imgsz=self.inferenceImgSize, verbose=self.isVerbose, device=self.device)

    def getDetectionFromResult(self, result, originalWH : Optional[Tuple[int,int]]) -> sv.Detections:
        if originalWH is None:
            originalWH = (640,640)

        det = sv.Detections.from_ultralytics(result)

        return self._remapDetectionsIfNeeded(det, originalWH)


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
            originalWH = (640,640)

        det = sv.Detections.from_inference(result)

        return self._remapDetectionsIfNeeded(det, originalWH)
