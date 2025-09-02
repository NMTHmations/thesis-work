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
    modelL = DetectionModelFactory.create(**modelConfig)
    modelR = DetectionModelFactory.create(**modelConfig)

    frameBufferL = FrameBuffer(maxLength=256)
    frameBufferR = FrameBuffer(maxLength=256)

    frameBufferL.timeout = 0.15
    frameBufferR.timeout = 0.15

    stopEvent = threading.Event()

    detectionQueueL = queue.Queue(maxsize=64)
    detectionQueueR = queue.Queue(maxsize=64)

    threads = (
        #TODO(IMPLEMENT)
    )

    threadManager = ThreadManager(stopEvent=stopEvent, threads=threads)
    threadManager.start()
    threadManager.join()

if __name__ == "__main__":
    main()
