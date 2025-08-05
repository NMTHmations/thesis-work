from ultralytics import YOLO
import cv2

"""
    Models are and should be in the project/models folder 
"""



class BallDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def processSingleFrame(self, frame):
        """
            Processes a single frame:
            - Converts it to grayscale
            - Runs YOLO to detect sports balls (class ID 32)
            - Returns both grayscale and detection overlay frames
        """

        """ convert to grayscale """
        grayscaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        """class 32 -> sports ball"""
        # result = model(frame, stream=True, classes=32)

        result = self.model.track(frame, tracker="bytetrack.yaml", classes=32)
        #result = self.model.predict(frame, tracker="bytetrack.yaml", classes=32)

        """ end result """
        results = list(result)

        """If any object were detected, draw it on the frame"""
        if results:
            detectedObjects = results[0].plot()
        else:
            detectedObjects = frame

        return grayscaleFrame, detectedObjects

    def processVideo(self, path: str = None):
        """
            - Captures video from webcam and processes each frame.
            - Yields original, grayscale, and detection frames.

            In case testing with video file,
             the path should be given in the following format: "project/sources/vid/testVid.fileFormat"
        """

        if path:
            capture = cv2.VideoCapture(path)
        else:
            capture = cv2.VideoCapture(0)

        if not capture.isOpened():
            raise IOError("Cannot capture/open video.")

        try:
            while True:
                # Read frames
                ret, frame = capture.read()
                if not ret:
                    break

                # frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

                graysclafeFrame, detectedObjects = self.processSingleFrame(frame)

                yield frame, graysclafeFrame, detectedObjects

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            capture.release()
