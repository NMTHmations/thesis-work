import threading
import time

import cv2

from project.detection.types.FrameBuffer import FrameBuffer
from project.detection.types.FrameItem import FrameItem
from project.detection.types.enums import ThreadNames


class CaptureThread(threading.Thread):
    def __init__(self, stopEvent : threading.Event, source : int | str, frameBuffer : FrameBuffer):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        self.frameBuffer : FrameBuffer = frameBuffer
        self.stopEvent = stopEvent
        self.frameID = 0

    def run(self):

        if not self.cap.isOpened():
            print(f"[{ThreadNames.CAPTURE}] Could not open camera")
            self.stopEvent.set()
            return

        while not self.stopEvent.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print(f"[{ThreadNames.CAPTURE}] Could not read frame (end of video or camera error)")
                break

            self.frameID += 1
            item = FrameItem(frameID=self.frameID, frame=frame, timestamp=time.time())
            self.frameBuffer.push(item)

        self.cap.release()
        print(f"[{ThreadNames.CAPTURE}]  Thread stopped.")




