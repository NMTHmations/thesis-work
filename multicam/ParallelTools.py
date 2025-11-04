import time
import cv2
import numpy as np
from . import DetermineStrike
from multiprocessing import Process, Event
from multiprocessing import Value, Queue, Manager
from picamera2.devices import Hailo
from . import MotorClient
from project.detection.types.ODModel import ColorDetectorModel


class ParallelTools():

    def __init__(self,strikeFront: dict, camFront:str|int,camDexter:str|int = "", strikeDexter:dict = {}, albument: bool = False, startStep = 0, endStep = 600, debug: bool = False):
        self.sourceFront = camFront
        self.sourceDexter = camDexter
        self.strikeFront = strikeFront
        self.strikeDexter = strikeDexter
        self.InputFront = Queue()
        self.InputSide = Queue()
        self.OutputSide = Queue()
        self.OutputFront = Queue()
        self.MotorController = MotorClient.MotorClient()
        self.startStep = startStep
        self.endStep = endStep
        self.albument = albument
        self.lowerHSV = (45, 167, 0)
        self.upperHSV = (89, 255, 255)
        self.debug = debug
    
    def _cameraHandler(self,source,strikeEstimaterDetails,start_event,stop_event, isSide:bool, isGoalFront, isGoalDexter):
        start_event.wait()
        strikeEstimaterDetails["debug"] = self.debug
        strikeEstimater = DetermineStrike.DetermineStrike(**strikeEstimaterDetails)
        cap = cv2.VideoCapture(source)
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
            input_data = np.expand_dims(frame.astype(np.uint8), 0) if self.debug == False else frame
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

    def getGoalResults(self,start_event,stop_event, isGoalFront, isGoalDexter, currentStep, strikeFront):
        start_event.wait()
        while not stop_event.is_set():
            if isGoalFront.value is not None and isGoalDexter.value is not None:
                print("Goal!")
                maxStep = self.endStep - self.startStep
                ratio = maxStep / (strikeFront["acceptEnd"][0] - strikeFront["acceptStart"][0])
                pointX = min(self.startStep + round(isGoalFront.value[0] * ratio), self.endStep)
                steps = pointX - currentStep.value
                direction = True
                if steps < 0:
                    direction = False
                for i in range(abs(steps)):
                    self.MotorController.stepMotor(direction)
                currentStep.value = pointX
                isGoalFront.value = None
                isGoalDexter.value = None
    
    def InferenceJudge(self,start_event,stop_event):
        start_event.wait()
        Model = None
        if self.debug == False:
            Model = Hailo("hailort/ball-detection--640x480_quant_hailort_hailo8_1.hef")
        else:
            Model = ColorDetectorModel(self.lowerHSV, self.upperHSV)
        while not stop_event.is_set():
            try:
                ts_front, input_front = self.InputFront.get_nowait()
                result = None
                if self.debug == False:
                    output = Model.run_async(input_front)
                    result = output.result()
                else:
                    result = Model.infer(input_front)
                self.OutputFront.put((ts_front, result))
            except:
                pass
            
            try:
                ts_side, input_side = self.InputSide.get_nowait()
                output = Model.run_async(input_side)
                result = output.result()
                self.OutputSide.put((ts_side, result))
            except:
                pass

        if self.debug == False:    
            Model.close()

    def CameraHandler(self):
        start_event = Event()
        stop_event = Event()
        currentStep = Value("i",self.startStep)
        manager = Manager()
        isGoalFront = manager.Value(object, None)
        isGoalDexter = manager.Value(object, None) if self.debug == False else manager.Value(object, True)
        processFront = Process(target=self._cameraHandler,args=(self.sourceFront, self.strikeFront,start_event,stop_event,False, isGoalFront, isGoalDexter))
        if self.debug == False:
            processDexter = Process(target=self._cameraHandler,args=(self.sourceDexter, self.strikeDexter,start_event,stop_event,True, isGoalFront, isGoalDexter))
        goalSummarizer = Process(target=self.getGoalResults,args=(start_event,stop_event, isGoalFront, isGoalDexter, currentStep, self.strikeFront))
        hailoGod = Process(target=self.InferenceJudge,args=(start_event,stop_event))
        processFront.start()
        if self.debug == False:
            processDexter.start()
        goalSummarizer.start()
        hailoGod.start()
        start_event.set()
        processFront.join()
        if self.debug == False:
            processDexter.join()
        goalSummarizer.join()
        hailoGod.join()

