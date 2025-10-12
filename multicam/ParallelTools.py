import time
import cv2
import numpy as np
from DetermineStrike import DetermineStrike
from multiprocessing import Process, Event
from multiprocessing import Value, Queue, Manager
from picamera2.devices import Hailo
from MotorClient import MotorClient


class ParallelTools():
    
    
    def __init__(self,camFront:str|int,camDexter:str|int, strikeFront, strikeDexter, startingStep : int = 0, maxStep :int = 30, albument: bool = False):
        self.sourceFront = camFront
        self.sourceDexter = camDexter
        self.strikeFront = strikeFront
        self.strikeDexter = strikeDexter
        self.InputFront = Queue()
        self.InputSide = Queue()
        self.OutputSide = Queue()
        self.OutputFront = Queue()
        self.startingPoint = startingStep
        self.MotorController = MotorClient()
        self.maxStep = maxStep
        self.albument = albument
    
    def _cameraHandler(self,source,strikeEstimaterDetails,start_event,stop_event, isSide:bool, isGoalFront, isGoalDexter):
        start_event.wait()
        strikeEstimater = DetermineStrike(**strikeEstimaterDetails)
        cap = cv2.VideoCapture(filename=source)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FPS,120)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print(cap.get(cv2.CAP_PROP_FPS))
        is_goal = False
        detection_buffer = []
        non_detection_counter = 0

        while not stop_event.is_set():
            actual_frame = None
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
                if abs(ts - timestamp) < 0.15:
                    matched_detections = dets
                    break
                
            if matched_detections is not None:
                annotated_frame, is_goal = strikeEstimater.detectFrame(frame, matched_detections, self.albument)
                if isSide:
                    isGoalDexter.value = is_goal
                else:
                    isGoalFront.value = is_goal
                actual_frame = annotated_frame
            else:
                non_detection_counter += 1
                if non_detection_counter == 10:
                    strikeEstimater.flushPositions()
                    non_detection_counter = 0
                actual_frame = frame
            
            cv2.imshow("YOLOv5 Detection", actual_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        cap.release()
        cv2.destroyAllWindows()
        exit()

    def getGoalResults(self,start_event,stop_event, isGoalFront, isGoalDexter, startStep):
        start_event.wait()
        while not stop_event.is_set():
            if isGoalFront.value is not None and isGoalDexter.value is not None:
                print("Goal!")
                ratio = self.maxStep / 640
                pointX = round(isGoalFront.value[0] * ratio)
                steps = pointX - startStep.value
                direction = True
                if steps < 0:
                    direction = False
                for i in range(abs(steps)):
                    self.MotorController.stepMotor(direction)
                startStep.value = pointX
                isGoalFront.value = None
                isGoalDexter.value = None
    
    def HailoInferenceJudge(self,start_event,stop_event):
        start_event.wait()
        HailoModel = Hailo("../hailort/ball-detection--640x480_quant_hailort_hailo8_1.hef")
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
        startStep = Value("i",self.startingPoint)
        manager = Manager()
        isGoalFront = manager.Value(object, None)
        isGoalDexter = manager.Value(object, None)
        processFront = Process(target=self._cameraHandler,args=(self.sourceFront, self.strikeFront,start_event,stop_event,False, isGoalFront, isGoalDexter))
        processDexter = Process(target=self._cameraHandler,args=(self.sourceDexter, self.strikeDexter,start_event,stop_event,True, isGoalFront, isGoalDexter))
        goalSummarizer = Process(target=self.getGoalResults,args=(start_event,stop_event, isGoalFront, isGoalDexter, startStep))
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

