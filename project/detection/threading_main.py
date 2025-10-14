import queue
import threading

import numpy as np

from threads import *
from project.detection.types.FrameBuffer import FrameBuffer
from project.detection.types.ODModel import DetectionModelFactory
from project.detection.types.enums import ModelTypes, FrameSize


def main():
    source = 1
    # source = "../sources/vid/speed_example_720p.mp4"
    #modelPath = "../models/best.engine"
    # modelPath = "experiment-sxxxi/1"
    device = 0  # GPU: 0 vagy 'cuda:0', CPU: 'cpu'

    lowerHSV = np.array([40, 40, 40])
    upperHSV = np.array([80, 255, 255])

    modelConfig = {
        "modelType": ModelTypes.COLOR,
        "loverHSV" : lowerHSV,
        "upperHSV" : upperHSV,
    }
    model = DetectionModelFactory.create(**modelConfig)

    frameBuffer = FrameBuffer(maxLength=64)
    frameBuffer.timeout = 0.15

    stopEvent = threading.Event()

    detectionQueue = queue.Queue(maxsize=64)

    threads = (
        CaptureThread(stopEvent=stopEvent, source=source, frameBuffer=frameBuffer),
        DetectionThread(stopEvent=stopEvent, frameBuffer=frameBuffer, detectionQueue=detectionQueue, model=model, batchSize=1),
        VisualizerThread(stopEvent=stopEvent, detectionQueue=detectionQueue)
    )

    threadManager = ThreadManager(stopEvent=stopEvent, threads=threads)
    threadManager.start()
    threadManager.join()

if __name__ == "__main__":
    main()
