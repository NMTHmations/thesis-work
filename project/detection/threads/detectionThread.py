import queue
import threading
import traceback
from typing import List, Tuple, Optional

import cv2
import supervision as sv


from project.detection.types.FrameBuffer import FrameBuffer, FrameItem
from project.detection.types.ODModel import DetectionModel
from project.detection.types.enums import ThreadNames, FrameSize


class DetectionThread(threading.Thread):
    def __init__(
        self,
        stopEvent: threading.Event,
        frameBuffer : FrameBuffer,
        detectionQueue: queue.Queue,
        model : DetectionModel,
        batchSize: int = 5,
        inputDimension: int = FrameSize.INPUTDIM,
    ):
        super().__init__(daemon=True)
        self.stopEvent = stopEvent
        self.frameBuffer = frameBuffer
        self.detectionQueue = detectionQueue
        self.batchSize = batchSize
        self.inputDimension = inputDimension

        self.model = model

        self.boxAnnotator = sv.BoxAnnotator()
        self.labelAnnotator = sv.LabelAnnotator()

    def _preprocessFrames(self, frames : List[FrameItem], imgSize: Tuple[int, int]) -> List[FrameItem]:
        items = []
        for frameItem in frames:
            processed = cv2.resize(frameItem.frame, imgSize)
            items.append(processed)

        return items

    def run(self):
        print(f"[{ThreadNames.DETECTION}] Thread started")
        if self.model is None:
            print(f"[{ThreadNames.DETECTION}] No model loaded, exiting")
            return

        while not self.stopEvent.is_set():
            items = self.frameBuffer.pop_batch(self.batchSize)

            if not items:
                continue

            originalBatch = items
            batchImages = self._preprocessFrames(items, (self.inputDimension, self.inputDimension))

            try:
                results = self.model.batch_infer(batchImages)
            except Exception as e:
                print(f"[{ThreadNames.DETECTION}] Inference failed:", e)
                traceback.print_exc()
                continue

            for it, result in zip(originalBatch, results):
                orig_h, orig_w = it.frame.shape[:2]

                detections = self.model.getDetectionFromResult(result, originalWH=(orig_w,orig_h))

                annotated = it.frame.copy()
                try:
                    annotated = self.boxAnnotator.annotate(annotated, detections)
                    labels = [f"{c}:{round(s,2)}" for c, s in zip(detections.class_id, detections.confidence)]
                    annotated = self.labelAnnotator.annotate(annotated, detections, labels)
                except Exception as e:
                    print(f"[{ThreadNames.DETECTION}] Annotation failed:", e)
                    traceback.print_exc()
                    continue

                if not self.detectionQueue.full():
                    self.detectionQueue.put((it.frameID, annotated))

        print(f"[{ThreadNames.DETECTION}] Thread stopped")
