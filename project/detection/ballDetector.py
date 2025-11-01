import supervision as sv
from ultralytics import YOLO
import cv2
from inference.models.utils import get_model

"""
    Models are and should be in the project/models folder 
"""
class BallDetector():
    def processSingleFrame(self, frame):
        raise NotImplementedError()

    def processVideo(self, path: str = None):
        raise NotImplementedError()



#modes : sv, yolo
class BD():
    def __init__(self, model_path: str, mode: str = "yolo", api_key = None):
        self.mode = mode

        print(f"Mode: {self.mode}")
        if mode == "yolo":
            self.model = YOLO(model_path)
            print("YOLO model loaded")
        elif mode == "sv":
            self.model = get_model(model_path, api_key=api_key)
            print("SV model loaded")
        else:
            raise ValueError("Unknown mode\nMode must be 'yolo' or 'sv'")

    def processVideo(self, path: str = None):
        if self.mode == "yolo":
            pass
        elif self.mode == "sv":
            pass


    def __processYOLO(self, path: str = None):
        pass

    def __processSV(self, path: str = None):
        pass

    def __processSingleFrameYOLO(self, frame):
        pass

    def __processSingleFrameSV(self, frame):
        pass



class BallDetector_SV_RF(BallDetector):
    def __init__(self, model_path):
        self.model = get_model(model_path, api_key="PlEVRUdW9e6KwDkUHIX6")
        print(f"Model loaded {model_path} <Supervision>")
        self.boxAnnotator = sv.BoxAnnotator()
        self.labelAnnotator = sv.LabelAnnotator()



    def processSingleFrame(self, frame):

        results = self.model.infer(frame)[0]
        detections = sv.Detections.from_inference(results)

        grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        annotated_frame = self.boxAnnotator.annotate(frame, detections)

        return annotated_frame, grayscale


    def processVideo(self, path: str = None):


        if path:
            frames = sv.get_video_frames_generator(path)
        else:
            capture = cv2.VideoCapture(0)

            def get_camera_frame_generator():
                try:
                    while True:
                        ret, frame = capture.read()

                        if not ret:
                            break

                        yield frame

                finally:
                    capture.release()

            frames = get_camera_frame_generator()

        for f in frames:
            annotated_frame, grayscale = self.processSingleFrame(f)
            yield f, grayscale, annotated_frame


class BallDetector_YOLO_CV2(BallDetector):
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
        result = self.model(frame, stream=True, classes=32)

        #result = self.model.track(frame, tracker="bytetrack.yaml", classes=32)
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

                frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

                graysclafeFrame, detectedObjects = self.processSingleFrame(frame)

                yield frame, graysclafeFrame, detectedObjects

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            capture.release()
