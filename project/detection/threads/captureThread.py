import queue
import threading

import cv2


class CaptureThread(threading.Thread):
    def __init__(self, stopEvent : threading.Event, source : int | str, frameQueue : queue.Queue):
        super().__init__()
        self.cap = cv2.VideoCapture(source)
        self.frameQueue = frameQueue
        self.stopEvent = stopEvent

    def run(self):
        while not self.stopEvent.is_set():
            ret, frame = self.cap.read()
            if not ret:
                break

            if not self.frameQueue.full():
                self.frameQueue.put(frame)

        self.cap.release()




