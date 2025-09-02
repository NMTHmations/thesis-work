import queue
import threading

import cv2

from project.detection.types.enums import ThreadNames


class VisualizerThread(threading.Thread):
    def __init__(self, stopEvent, detectionQueue, windowName="Display"):
        super().__init__(daemon=True)
        self.stopEvent = stopEvent
        self.detectionQueue = detectionQueue

        self.windowName = windowName + str(self.__hash__())


    def run(self):
        print(f"[{ThreadNames.VISUALIZER}] Thread started")
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 1280, 720)

        while not self.stopEvent.is_set():
            try:
                frame_id, frame = self.detectionQueue.get()
            except queue.Empty:
                print(f"[{ThreadNames.VISUALIZER}] No frames received")
                continue

            cv2.imshow(self.windowName, frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopEvent.set()
                break

        cv2.destroyAllWindows()
        print(f"[{ThreadNames.VISUALIZER}] Thread stopped")
