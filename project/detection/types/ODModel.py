from abc import abstractmethod

from inference import get_model, Model


class DetectionModel:
    @abstractmethod
    def infer(self):
        pass

    @abstractmethod
    def batch_infer(self):
        pass

class DetectionModelFactory:
    @staticmethod
    def create(**kwargs) -> DetectionModel:
        pass




class YOLOModel(DetectionModel):
    def __init__(self, modelPath : str):
        from ultralytics import YOLO

        self.model = YOLO(modelPath)

    def infer(self):
        pass

    def batch_infer(self):
        pass


class RoboflowModel(DetectionModel):
    def __init__(self, modelPath : str, apiKey : str):
        self.model : Model = get_model(modelPath, api_key=apiKey)

    def infer(self):
        pass

    def batch_infer(self):
        pass
