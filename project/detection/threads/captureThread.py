import threading
import time

import cv2
from project.detection.types.FrameBuffer import FrameBuffer, FrameItem
from project.detection.types.enums import ThreadNames


class CaptureThread(threading.Thread):
    """
    A dedicated thread to capture frames from a video source.
    Pushing them into a shared FrameBuffer.

    """
    def __init__(
            self,
            stopEvent: threading.Event,
            source: int | str,
            frameBuffer: FrameBuffer
    ):

        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(source)
        self.frameBuffer = frameBuffer
        self.stopEvent = stopEvent
        self.frameID = 0


    def run(self):
        print(f"[{ThreadNames.CAPTURE}] Thread started")

        if not self.cap.isOpened():
            print(f"[{ThreadNames.CAPTURE}] Cannot open source")
            self.stopEvent.set()
            return

        #Main loop for frame capturing
        while not self.stopEvent.is_set():

            ret, frame = self.cap.read()
            if not ret:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frameID = 0
                continue
                """print(f"[{ThreadNames.CAPTURE}] No frame (end of video or camera error).")
                break"""

            self.frameID += 1

            item = FrameItem(frameID=self.frameID, frame=frame, timestamp=time.time())
            self.frameBuffer.push(item)

            # Optional: Uncomment to throttle capture speed if FPS is too high
            # time.sleep(0.001)

        self.cap.release()
        print(f"[{ThreadNames.CAPTURE}] Thread stopped")