from abc import abstractmethod
import supervision as sv
from inference import get_model, Model


class DetectionModel:
    @abstractmethod
    def infer(self, frame):
        pass

    @abstractmethod
    def batch_infer(self, batch):
        pass

    @abstractmethod
    def getDetectionFromResult(self) -> sv.Detections:
        pass

class DetectionModelFactory:
    @staticmethod
    def create(**kwargs) -> DetectionModel:
        pass




class YOLOModel(DetectionModel):
    def __init__(self, modelPath : str):
        from ultralytics import YOLO

        self.model = YOLO(modelPath)

    def infer(self, frame):
        pass

    def batch_infer(self, batch):
        pass

    def getDetectionFromResult(self) -> sv.Detections:
        pass

    def _mapDetectionsToOriginal(self):
        #From detection thread
        pass

class RoboflowModel(DetectionModel):
    def __init__(self, modelPath : str, apiKey : str):
        self.model : Model = get_model(modelPath, api_key=apiKey)

    def infer(self, frame):
        pass

    def batch_infer(self, batch):
        pass

    def getDetectionFromResult(self) -> sv.Detections:
        pass


    def _mapDetectionsToOriginal(self):
        pass

