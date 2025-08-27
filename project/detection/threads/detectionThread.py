import queue
import threading
from typing import List, Tuple, Optional

import cv2
from inference import get_model
from supervision.annotators.base import BaseAnnotator
from ultralytics import YOLO
import supervision as sv

from project.detection.types.FrameBuffer import FrameBuffer
from project.detection.types.FrameItem import FrameItem
from project.detection.types.ODModel import DetectionModel
from project.detection.types.enums import ThreadNames


class DetectionThread_YOLO(threading.Thread):
    def __init__(self, stopEvent: threading.Event, frameQueue: queue.Queue, detectionQueue: queue.Queue, model):
        super().__init__()
        self.frameQueue = frameQueue
        self.detectionQueue = detectionQueue
        self.stopEvent = stopEvent

        self.model = model

    def run(self):
        while not self.stopEvent.is_set():
            if not self.frameQueue.empty():
                frame = self.frameQueue.get()

                results = self.model.track(frame)[0]
                detection = sv.Detections.from_ultralytics(results)


                if not self.detectionQueue.full():
                    self.detectionQueue.put(detection)

class DetectionThread(threading.Thread):
    def __init__(
            self,
            stopEvent : threading.Event,
            frameBuffer : FrameBuffer,
            detectionQueue : queue.Queue,
            model : DetectionModel,
            originalFrameSize : Tuple[int, int],
            annotators : Optional[List[BaseAnnotator]] | Optional[Tuple[BaseAnnotator]] | Optional[BaseAnnotator],
            batchSize : int = 5,
    ):

        super().__init__()
        self.stopEvent = stopEvent
        self.frameBuffer = frameBuffer
        self.detectionQueue = detectionQueue
        self.batchSize = batchSize
        self.model = model

        #(Width,Height)
        self.originalFrameSize = originalFrameSize

        if annotators is None:
            self.annotators = sv.BoxAnnotator
        else:
            self.annotators = annotators

    def run(self):
        print(f"{ThreadNames.DETECTION} Started.")
        boxAnnotator = sv.BoxAnnotator()

        if self.model is None:
            print(f"[{ThreadNames.DETECTION}] No model loaded")
            self.stopEvent.set()
            return

        while not self.stopEvent.is_set():
            items = self.frameBuffer.pop_batch(self.batchSize)

            if not items:
                continue

            originalBatch = items
            preprocessedBatch = self._preprocessFrames(items)

            results = None
            try:
                results = self.model.batch_infer(preprocessedBatch)
            except Exception as e:
                print(f"[{ThreadNames.DETECTION}] Inference failed:", e)
                for original in originalBatch:
                    if not self.detectionQueue.full():
                        self.detectionQueue.put(original)
                continue

            for original, result in zip(originalBatch, results):
                detections = self.model.getDetectionFromResult(result, originalWH=self.originalFrameSize)

                annotated = original.frame.copy()

                annotated = boxAnnotator.annotate(annotated, detections)

                if not self.detectionQueue.full():
                    self.detectionQueue.put(annotated)

        print(f"[{ThreadNames.DETECTION}] Finished]")





    def _preprocessFrames(self, frames : List[FrameItem], imgSize: Tuple = (640,640)):
        items = []
        for frameItem in frames:
            processed = cv2.resize(frameItem.frame, imgSize)
            items.append(processed)

        return items