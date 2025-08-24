import queue
import threading
import supervision as sv

from threads import *

if __name__ == '__main__':
    # source = "../sources/vid/speed_example_720p.mp4"
    #source = "../sources/vid/real4.mp4"
    sources = ["../sources/vid/real4.mp4", "../sources/vid/real5.mp4"]
    # source = 0

    modelPath = "experiment-sxxxi/1"
    # modelPath = "../models/yolo11l.engine"

    frameQueue1 = queue.Queue()
    frameQueue2 = queue.Queue()
    detectionQueue1 = queue.Queue()
    detectionQueue2 = queue.Queue()

    stopEvent = threading.Event()

    annotators = (
        sv.BoxAnnotator(),
        sv.LabelAnnotator()
    )

    threads = (
        CaptureThread(stopEvent=stopEvent, source=sources[0], frameQueue=frameQueue1),
        CaptureThread(stopEvent=stopEvent, source=sources[1],frameQueue=frameQueue2),
        DetectionThread(stopEvent=stopEvent, detectionQueue=detectionQueue1, frameQueue=frameQueue1, modelPath=modelPath),
        DetectionThread(stopEvent=stopEvent, detectionQueue=detectionQueue2, frameQueue=frameQueue2, modelPath=modelPath),
        VisualizerThread(stopEvent=stopEvent, frameQueue=frameQueue1, detectionQueue=detectionQueue1, annotators=annotators),
        VisualizerThread(stopEvent=stopEvent, frameQueue=frameQueue2, detectionQueue=detectionQueue2, annotators=annotators),
    )

    pipeline = ThreadManager(threads=threads, queues=(frameQueue1, frameQueue2, detectionQueue1, detectionQueue2))

    pipeline.start()
    pipeline.join()
