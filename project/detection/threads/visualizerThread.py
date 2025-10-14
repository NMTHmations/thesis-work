import queue
import threading

import cv2

from project.detection.types.enums import ThreadNames


class VisualizerThread(threading.Thread):
    """
    A dedicated thread to display the processed video frames.
    """
    def __init__(
            self,
            stopEvent : threading.Event,
            detectionQueue : queue.Queue,
            windowName : str ="Display"
    ):

        super().__init__(daemon=True)
        self.stopEvent = stopEvent
        self.detectionQueue = detectionQueue

        self.windowName = windowName + str(self.__hash__())


    def run(self):
        print(f"[{ThreadNames.VISUALIZER}] Thread started")
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 640, 360)

        while not self.stopEvent.is_set():
            try:
                frameID, frame = self.detectionQueue.get()
            except queue.Empty:
                print(f"[{ThreadNames.VISUALIZER}] No frames received")
                continue

            cv2.resize(frame, (640,360))
            cv2.putText(frame,str(frameID),[100,100],  cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow(self.windowName, frame)


            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stopEvent.set()
                break

        cv2.destroyAllWindows()
        print(f"[{ThreadNames.VISUALIZER}] Thread stopped")
