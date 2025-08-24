import queue
import threading

from inference import get_model
from ultralytics import YOLO
import supervision as sv

class DetectionThread_YOLO(threading.Thread):
    def __init__(self, stopEvent: threading.Event, frameQueue: queue.Queue, detectionQueue: queue.Queue,modelPath: str):
        super().__init__()
        self.frameQueue = frameQueue
        self.detectionQueue = detectionQueue
        self.stopEvent = stopEvent

        self.model = YOLO(modelPath)

    def run(self):
        while not self.stopEvent.is_set():
            if not self.frameQueue.empty():
                frame = self.frameQueue.get()

                results = self.model.track(frame)[0]
                detection = sv.Detections.from_ultralytics(results)


                if not self.detectionQueue.full():
                    self.detectionQueue.put(detection)

class DetectionThread(threading.Thread):
    def __init__(self, stopEvent : threading.Event, frameQueue : queue.Queue, detectionQueue : queue.Queue, modelPath: str):
        super().__init__()
        self.frameQueue = frameQueue
        self.detectionQueue = detectionQueue
        self.stopEvent = stopEvent

        self.model = get_model(modelPath, api_key="PlEVRUdW9e6KwDkUHIX6")

    def run(self):
        while not self.stopEvent.is_set():
            if not self.frameQueue.empty():
                frame = self.frameQueue.get()
                results = self.model.infer(frame)[0]
                detection = sv.Detections.from_inference(results)

                if not self.detectionQueue.full():
                    self.detectionQueue.put(detection)
