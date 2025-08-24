import threading
import time

import supervision as sv
import cv2


class VisualizerThread(threading.Thread):
    def __init__(self, stopEvent : threading.Event, detectionQueue, annotators : tuple):
        super().__init__()
        self.detectionQueue = detectionQueue
        self.annotators = annotators
        self.stopEvent = stopEvent


    def run(self):
        winName = f'Annotated Frame [{time.time()}]'
        cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(winName, 1280, 720)

        while not self.stopEvent.is_set():
            if not self.detectionQueue.empty():
                frame, detections = self.detectionQueue.get()

                annotatedFrame = frame.copy()

                for annotator in self.annotators:
                    if isinstance(annotator, sv.LabelAnnotator):
                        labels = [f"#{classID}" for classID in detections.class_id]
                        annotatedFrame = annotator.annotate(frame, detections, labels)
                    else:
                        annotatedFrame = annotator.annotate(frame, detections)


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stopEvent.set()


                cv2.imshow(winName, annotatedFrame)

        cv2.destroyAllWindows()