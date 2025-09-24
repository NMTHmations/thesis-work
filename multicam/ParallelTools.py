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
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        is_goal = False
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                #strikeEstimater.flushPositions()
                continue
            frame = cv2.resize(frame, (640, 480))
            input_data = frame.astype(np.uint8)
            input_data = np.expand_dims(input_data, 0)
            if isSide:
                self.InputSide.put(input_data)
            else:
                self.InputFront.put(input_data)
            detections = None
            if isSide:
                if not self.OutputSide.empty():
                    detections = self.OutputSide.get()
            else:
                if not self.OutputFront.empty():
                    detections = self.OutputFront.get()
            if detections != None:
                frame, is_goal = strikeEstimater.detectFrame(frame,detections,False)
            if isSide:
                isGoalDexter.value = is_goal
            else:
                isGoalFront.value = is_goal
            if not ret:
                print("Error: Could not read frame.")
                break
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
            if not self.InputFront.empty():
                input = self.InputFront.get()
                output = HailoModel.run_async(input)
                result = output.result()
                self.OutputFront.put(result)
            if not self.InputSide.empty():
                input = self.InputSide.get()
                output = HailoModel.run_async(input)
                result = output.result()
                self.OutputSide.put(result)
        if stop_event.is_set():
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
    parallel = ParallelTools("multicam/real4.mp4","multicam/real4.mp4",dexterStrike,dexterStrike)
    parallel.CameraHandler()

