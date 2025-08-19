import threading
import supervision as sv
import cv2


class VisualizerThread(threading.Thread):
    def __init__(self, stopEvent : threading.Event,frameQueue, detectionQueue, annotators : tuple):
        super().__init__()
        self.frameQueue = frameQueue
        self.detectionQueue = detectionQueue
        self.annotators = annotators
        self.stopEvent = stopEvent


    def run(self):
        while not self.stopEvent.is_set():
            if not self.frameQueue.empty():
                frame = self.frameQueue.get()
                detections = self.detectionQueue.get()

                annotatedFrame = frame

                for annotator in self.annotators:
                    if isinstance(annotator, sv.LabelAnnotator):
                        labels = [f"#{classID}" for classID in detections.class_id]
                        annotatedFrame = annotator.annotate(frame, detections, labels)
                    else:
                        annotatedFrame = annotator.annotate(frame, detections)

                winName = 'Annotated Frame'

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stopEvent.set()

                cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(winName, 1280, 720)
                cv2.imshow(winName, annotatedFrame)
