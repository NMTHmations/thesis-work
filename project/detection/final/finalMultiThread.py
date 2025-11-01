import json
import threading

from supervision import BoxAnnotator

from project.detection.final.predictor import KFPredictor
from project.detection.final.singleThread import SingleThread
from project.detection.threads import ThreadManager
from project.detection.types.Camera import Camera
from project.detection.types.ODModel import ColorDetectorModel
from project.detection.types.Window import Window


def loadPoints(fpath):
    with open(fpath, "r") as f:
        return json.load(f)

def main():

    stopEvent = threading.Event()
    cam1 = Camera(1,captureWidth=1280, captureHeight=720, fps=60)
    cam2 = Camera(0,captureWidth=1280, captureHeight=720, fps=60)

    dt = 60.0

    window1 = Window("Cam1", 896,504)
    window2 = Window("Cam2", 896,504)

    # --- detektor és KF inicializálás ---
    lowerHSV = (45, 167, 0)
    upperHSV = (89, 255, 255)

    detector1 = ColorDetectorModel(lowerHSV, upperHSV)
    detector2 = ColorDetectorModel(lowerHSV, upperHSV)

    boxAnnotator1 = BoxAnnotator()
    boxAnnotator2 = BoxAnnotator()

    kf1 = KFPredictor(dimensions=2, dt=dt)
    kf2 = KFPredictor(dimensions=2, dt=dt)

    points1 = loadPoints("pointsFront.json")
    points2 = loadPoints("pointsSide.json")

    goalLine1 = (points1["GOAL1"], points1["GOAL2"])
    goalLine2 = (points2["GOAL1"], points2["GOAL2"])

    s1 = SingleThread(cam1,window1,detector1,kf1,goalLine1,stopEvent)
    s2 = SingleThread(cam2,window2,detector2,kf2,goalLine2,stopEvent)

    threadManager = ThreadManager(stopEvent, (s1,s2))

    threadManager.start()
    threadManager.join()


if __name__ == '__main__':
    main()
