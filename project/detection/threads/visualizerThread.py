import threading
import time

import supervision as sv
import cv2

from project.detection.types.enums import ThreadNames


class VisualizerThread(threading.Thread):
    def __init__(self, stopEvent : threading.Event, detectionQueue):
        super().__init__()
        self.detectionQueue = detectionQueue
        self.stopEvent = stopEvent

    def _calculateFPS(self):
        pass

    def run(self):
        print(f"[{ThreadNames.VISUALIZER}] Started.")
        winName = f'Display [{time.time()}]'
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, 1280, 720)

        while not self.stopEvent.is_set():
            if not self.detectionQueue.empty():
                frame = self.detectionQueue.get(timeout=0.5)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stopEvent.set()
                    break


                cv2.imshow(winName, frame)

        cv2.destroyAllWindows()