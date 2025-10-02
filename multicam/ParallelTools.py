from collections import defaultdict, deque
import time
import cv2
import hailo_platform
import numpy as np
import torch
from DetermineStrike import DetermineStrike
from multiprocessing import Process, Event
from multiprocessing import Value, Queue
from picamera2.devices import Hailo
from hailo_platform.pyhailort.pyhailort import VDevice, HailoRTException


class ParallelTools():
    
    
    def __init__(self,camFront:str|int,camDexter:str|int, strikeFront, strikeDexter):
        self.sourceFront = camFront
        self.sourceDexter = camDexter
        self.strikeFront = strikeFront
        self.strikeDexter = strikeDexter
        self.InputFront = Queue()
        self.InputSide = Queue()
        self.OutputSide = Queue()
        self.OutputFront = Queue()
    
    def _setCaptures(self, cap):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        coordinates = defaultdict(lambda: deque(maxlen=int(cap.get(cv2.CAP_PROP_FPS))))
        return cap, coordinates
    
    def _cameraHandler(self,source,strikeEstimaterDetails,start_event,stop_event, isSide:bool, isGoalFront, isGoalDexter):
        start_event.wait()
        strikeEstimater = DetermineStrike(**strikeEstimaterDetails)
        #cap, coord = self._setCaptures(cv2.VideoCapture(filename=source))
        cap = cv2.VideoCapture(filename=source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS,120)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(cap.get(cv2.CAP_PROP_FPS))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_goal = False
        previous_frame = None
        previous_detections = None
        detection_buffer = []

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            
            frame = cv2.resize(frame, (640, 480))
            input_data = np.expand_dims(frame.astype(np.uint8), 0)
            timestamp = time.time()

            if isSide:
                self.InputSide.put((timestamp, input_data))
            else:
                self.InputFront.put((timestamp, input_data))

            try:
                if isSide:
                    while True:
                        ts, dets = self.OutputSide.get_nowait()
                        detection_buffer.append((ts, dets))
                else:
                    while True:
                        ts, dets = self.OutputFront.get_nowait()
                        detection_buffer.append((ts, dets))
            except:
                pass
            
            matched_detections = None
            for ts, dets in reversed(detection_buffer):
                if abs(ts - timestamp) < 0.05:
                    matched_detections = dets
                    break
                
            if matched_detections is not None:
                annotated_frame, is_goal = strikeEstimater.detectFrame(frame, matched_detections, False)
                if isSide:
                    isGoalDexter.value = is_goal
                else:
                    isGoalFront.value = is_goal
                cv2.imshow("YOLOv5 Detection", annotated_frame)
            else:
                cv2.imshow("YOLOv5 Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        cap.release()
        cv2.destroyAllWindows()
        exit()

    def getGoalResults(self,start_event,stop_event, isGoalFront, isGoalDexter):
        start_event.wait()
        while not stop_event.is_set():
            if isGoalFront.value and isGoalDexter.value:
                print("Goal!")
                isGoalFront.value = False
                isGoalDexter.value = False
    
    def HailoInferenceJudge(self,start_event,stop_event):
        start_event.wait()
        HailoModel = Hailo("hailort/ball-detection--640x480_quant_hailort_hailo8_1.hef")
        while not stop_event.is_set():
            try:
                ts_front, input_front = self.InputFront.get_nowait()
                output = HailoModel.run_async(input_front)
                result = output.result()
                self.OutputFront.put((ts_front, result))
            except:
                pass
            
            try:
                ts_side, input_side = self.InputSide.get_nowait()
                output = HailoModel.run_async(input_side)
                result = output.result()
                self.OutputSide.put((ts_side, result))
            except:
                pass
            
        HailoModel.close()

    def CameraHandler(self):
        start_event = Event()
        stop_event = Event()
        isGoalFront = Value('b', False)
        isGoalDexter = Value('b', False)
        processFront = Process(target=self._cameraHandler,args=(self.sourceFront, self.strikeFront,start_event,stop_event,False, isGoalFront, isGoalDexter))
        processDexter = Process(target=self._cameraHandler,args=(self.sourceDexter, self.strikeDexter,start_event,stop_event,True, isGoalFront, isGoalDexter))
        goalSummarizer = Process(target=self.getGoalResults,args=(start_event,stop_event, isGoalFront, isGoalDexter))
        hailoGod = Process(target=self.HailoInferenceJudge,args=(start_event,stop_event))
        processFront.start()
        processDexter.start()
        goalSummarizer.start()
        hailoGod.start()
        start_event.set()
        processFront.join()
        processDexter.join()
        goalSummarizer.join()
        hailoGod.join()

dexterStrike = {
    "start": 50,
    "end": 540,
    "height": 480,
    "width": 640,
    "acceptStart": (150, 250),
    "acceptEnd": (150, 100),
    "lowerHSV": [10, 108, 28],
    "upperHSV": [17, 221, 224],
    "debug": False
}

if __name__ == "__main__":
    parallel = ParallelTools("/dev/video0","/dev/video2",dexterStrike,dexterStrike)
    parallel.CameraHandler()

