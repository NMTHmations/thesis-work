import threading

import cv2

from project.detection.final.predictor import KFPredictor
from project.detection.types.ODModel import DetectionModel
from project.detection.types.Window import Window
from project.detection.types.Camera import Camera


class SingleThread(threading.Thread):
    def __init__(self,
                 camera : Camera,
                 window : Window,
                 model : DetectionModel,
                 predictor : KFPredictor,
                 fieldPoints : tuple,
                 stopEvent : threading.Event

                 ):
        super().__init__(daemon=True)
        self.camera = camera
        self.window = window
        self.model = model
        self.predictor = predictor
        self.stopEvent = stopEvent
        self.fieldPoints = fieldPoints

    def run(self):
        print("T1")
        if not self.camera.camera.isOpened():
            self.stopEvent.set()

        while not self.stopEvent.is_set():
            success, frame = self.camera.capture()

            if not success:
                self.stopEvent.set()

            self.window.showFrame(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.stopEvent.set()

