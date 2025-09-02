import queue
import threading

from threads import *
from project.detection.types.FrameBuffer import FrameBuffer
from project.detection.types.ODModel import DetectionModelFactory
from project.detection.types.enums import ModelTypes, FrameSize


def main():
    source = 0  # vagy "../videos/test.mp4"
    # source = "../sources/vid/speed_example_720p.mp4"
    modelPath = "../models/yolo11l.engine"
    # modelPath = "experiment-sxxxi/1"
    device = 0  # GPU: 0 vagy 'cuda:0', CPU: 'cpu'

    modelConfig = {
        "modelPath": modelPath,
        "modelType": ModelTypes.YOLO,
        "device": device,
    }
    model = DetectionModelFactory.create(**modelConfig)

    frameBuffer = FrameBuffer(maxLength=256)
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
