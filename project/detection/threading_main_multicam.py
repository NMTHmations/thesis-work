import queue
import threading

from threads import *
from project.detection.types.FrameBuffer import FrameBuffer
from project.detection.types.ODModel import DetectionModelFactory
from project.detection.types.enums import ModelTypes, FrameSize


def main():
    sourceLeft = 1
    sourceRight = 2

    modelPath = "../models/best.engine"
    # modelPath = "experiment-sxxxi/1"
    device = 0  # GPU: 0 vagy 'cuda:0', CPU: 'cpu'

    modelConfig = {
        "modelPath": modelPath,
        "modelType": ModelTypes.YOLO,
        "device": device,
    }
    modelLeft = DetectionModelFactory.create(**modelConfig)
    modelRight = DetectionModelFactory.create(**modelConfig)

    frameBufferLeft = FrameBuffer(maxLength=64)
    frameBufferLeft.timeout = 0.15

    frameBufferRight = FrameBuffer(maxLength=64)
    frameBufferRight.timeout = 0.15

    stopEvent = threading.Event()

    detectionQueueLeft = queue.Queue(maxsize=64)
    detectionQueueRight = queue.Queue(maxsize=64)

    threads = (
        CaptureThread(stopEvent=stopEvent, source=sourceLeft, frameBuffer=frameBufferLeft),
        CaptureThread(stopEvent=stopEvent, source=sourceRight, frameBuffer=frameBufferRight),
        DetectionThread(stopEvent=stopEvent, frameBuffer=frameBufferLeft, detectionQueue=detectionQueueLeft, model=modelLeft,
                        batchSize=1),
        DetectionThread(stopEvent=stopEvent, frameBuffer=frameBufferRight, detectionQueue=detectionQueueRight, model=modelRight,
                        batchSize=1),
        VisualizerThread(stopEvent=stopEvent, detectionQueue=detectionQueueLeft),
        VisualizerThread(stopEvent=stopEvent, detectionQueue=detectionQueueRight),
    )

    threadManager = ThreadManager(stopEvent=stopEvent, threads=threads)
    threadManager.start()
    threadManager.join()

if __name__ == "__main__":
    main()
