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
    """
    A dedicated thread to perform object detection on video frames.
    """
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
        """
        Performs frame preprocessing to speed up object detection.
        :param frames: A list of frame items.
        :param imgSize: Target image size.
        :return: A list of resized frames.
        """
        items = []
        for frameItem in frames:
            #processed = cv2.resize(frameItem.frame, imgSize)
            items.append(frameItem.frame)

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
                except Exception as e:
                    print(f"[{ThreadNames.DETECTION}] Annotation failed:", e)
                    traceback.print_exc()
                    continue

                if not self.detectionQueue.full():
                    self.detectionQueue.put((it.frameID, annotated))

        print(f"[{ThreadNames.DETECTION}] Thread stopped")
