import queue
import threading

import supervision as sv
from inference import get_model
from ultralytics import YOLO

from threads import *


def main():
    #source = "../sources/vid/speed_example_720p.mp4"
    source = "../sources/vid/real4.mp4"
    #source = 0

    modelPathInference = "experiment-sxxxi/1"
    modelPathYOLO = "../models/yolo11l.engine"

    modelYOLO = YOLO(modelPathYOLO)
    modelInference = get_model(modelPathInference, api_key="PlEVRUdW9e6KwDkUHIX6")

    frameQueue = queue.Queue()
    detectionQueue = queue.Queue()

    queues = (frameQueue, detectionQueue)

    annotators = (
        sv.BoxAnnotator(),
        sv.LabelAnnotator()
    )

    stopEvent = threading.Event()

    threads = (
        CaptureThread(stopEvent=stopEvent,source=source, frameQueue=frameQueue),
        DetectionThread(stopEvent=stopEvent,detectionQueue=detectionQueue, frameQueue=frameQueue, model=modelInference),
        #DetectionThread_YOLO(stopEvent=stopEvent,detectionQueue=detectionQueue, frameQueue=frameQueue, model=modelYOLO),
        VisualizerThread(stopEvent=stopEvent, detectionQueue=detectionQueue, annotators=annotators)
    )


    threadmanager = ThreadManager(queues=queues, threads=threads)

    threadmanager.start()
    threadmanager.join()



if __name__ == '__main__':
    main()