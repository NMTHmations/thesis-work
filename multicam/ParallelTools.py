from collections import defaultdict, deque
import time
import cv2
import torch
from DetermineStrike import DetermineStrike
from multiprocessing import Process, Event
from multiprocessing import Value


class ParallelTools():
    
    
    def __init__(self,camFront:str|int,camDexter:str|int, strikeFront, strikeDexter):
        self.sourceFront = camFront
        self.sourceDexter = camDexter
        self.strikeFront = strikeFront
        self.strikeDexter = strikeDexter
    
    def _setCaptures(self, cap):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        coordinates = defaultdict(lambda: deque(maxlen=int(cap.get(cv2.CAP_PROP_FPS))))
        return cap, coordinates
    
    def _cameraHandler(self,source,strikeEstimaterDetails,start_event,stop_event, isSide:bool, isGoalFront, isGoalDexter):
        start_event.wait()
        strikeEstimater = DetermineStrike(**strikeEstimaterDetails)
        cap, coord = self._setCaptures(cv2.VideoCapture(filename=source))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES,0)
                strikeEstimater.flushPositions()
                continue
            frame = cv2.resize(frame, (640,384)) # resize frame
            if not ret:
                print("Error: Could not read frame.")
                break
            frame, result = strikeEstimater.detectFrame(frame) # run detection
            if isSide:
                isGoalDexter.value = bool(result)
            else:
                isGoalFront.value = bool(result)
            cv2.imshow("YOLOv5 Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                stop_event.set()
                break
        cap.release()
        cv2.destroyAllWindows()

    def getGoalResults(self,start_event,stop_event, isGoalFront, isGoalDexter):
        start_event.wait()
        while not stop_event.is_set():
            if isGoalFront.value and isGoalDexter.value:
                print("Goal!")
                isGoalFront.value = False
                isGoalDexter.value = False
    
    def CameraHandler(self):
        start_event = Event()
        stop_event = Event()
        isGoalFront = Value('b', False)
        isGoalDexter = Value('b', False)
        processFront = Process(target=self._cameraHandler,args=(self.sourceFront, self.strikeFront,start_event,stop_event,False, isGoalFront, isGoalDexter))
        processDexter = Process(target=self._cameraHandler,args=(self.sourceDexter, self.strikeDexter,start_event,stop_event,True, isGoalFront, isGoalDexter))
        goalSummarizer = Process(target=self.getGoalResults,args=(start_event,stop_event, isGoalFront, isGoalDexter))
        processFront.start()
        processDexter.start()
        goalSummarizer.start()
        start_event.set()
        processFront.join()
        processDexter.join()
        goalSummarizer.join()

dexterStrike = {
    "start": 50,
    "end": 540,
    "acceptStart": (150, 250),
    "acceptEnd": (150, 100),
    "lowerHSV": [10, 108, 28],
    "upperHSV": [17, 221, 224],
    "debug": False
}

if __name__ == "__main__":
    parallel = ParallelTools("multicam/real4.mp4","multicam/real4.mp4",dexterStrike,dexterStrike)
    parallel.CameraHandler()

