import queue
import threading

import supervision as sv
from threads import *


def main():
    source = "../sources/vid/speed_example_720p.mp4"
    #source = 0
    modelPath = "experiment-sxxxi/1"

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
        DetectionThread(stopEvent=stopEvent,detectionQueue=detectionQueue, frameQueue=frameQueue, modelPath=modelPath),
        VisualizerThread(stopEvent=stopEvent,frameQueue=frameQueue, detectionQueue=detectionQueue, annotators=annotators),
    )


    threadmanager = ThreadManager(queues=queues, threads=threads)

    threadmanager.start()
    threadmanager.join()



if __name__ == '__main__':
    main()