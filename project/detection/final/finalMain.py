#!/usr/bin/env python3
import cv2
import numpy as np
import threading
import queue
import json
import supervision as sv
from argparse import ArgumentParser

from project.detection.final.predictor import KFPredictor
from project.detection.final.visualizeUtils import createVisualizeFrame
from project.detection.types.Camera import Camera
from project.detection.types.FrameBuffer import FrameBuffer
from project.detection.types.FrameItem import FrameItem
from project.detection.types.ODModel import ColorDetectorModel
from project.detection.types.Window import Window


# ----------------------------
# Kamera és ablak segédosztályok
# ----------------------------


# ----------------------------
# Segédfüggvények
# ----------------------------
def argsParser():
    parser = ArgumentParser()
    parser.add_argument("--camFront", type=int, default=0, help="Front Kamera index")
    parser.add_argument("--camSide", type=int, default=1, help="Side Kamera index")
    parser.add_argument("--fps", type=float, default=60.0, help="Kamera FPS")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def loadPoints(fpath):
    with open(fpath, "r") as f:
        return json.load(f)


def getCenter(positionXYXY: np.ndarray):
    x1, y1, x2, y2 = positionXYXY
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


# ----------------------------
# Main
# ----------------------------
def main(args):
    # --- kamerák inicializálása ---
    cameraFront = Camera(args.camFront, args.width, args.height, args.fps)
    cameraSide = Camera(args.camSide, args.width, args.height, args.fps)

    dt = 1.0 / args.fps if args.fps > 0 else 1 / 30.0

    # --- megjelenítő ablakok ---
    frontWindow = Window("Front Preview", 896, 504)
    sideWindow = Window("Side Preview", 896, 504)
    goalWindow = Window("Goal Preview", 640, 360)

    # --- pontok betöltése JSON-ból ---
    pointsFront = loadPoints("pointsFront.json")
    pointsSide = loadPoints("pointsSide.json")

    goalLineFront = (pointsFront["GOAL1"], pointsFront["GOAL2"])
    goalLineSide = (pointsSide["GOAL1"], pointsSide["GOAL2"])

    print(f"[INFO] Front goal line: {goalLineFront}")
    print(f"[INFO] Side goal line: {goalLineSide}")

    # --- detektor és KF inicializálás ---
    lowerHSV = (45, 167, 0)
    upperHSV = (89, 255, 255)

    detectorFront = ColorDetectorModel(lowerHSV, upperHSV)
    detectorSide = ColorDetectorModel(lowerHSV, upperHSV)



    boxAnnotator = sv.BoxAnnotator()

    kfFront = KFPredictor(dimensions=2, dt=dt)
    kfSide = KFPredictor(dimensions=2, dt=dt)

    stopEvent = threading.Event()
    counter = 0

    while not stopEvent.is_set():
        goalframe = createVisualizeFrame()

        successFront, frameFront = cameraFront.capture()
        successSide, frameSide = cameraSide.capture()
        if not successFront or not successSide:
            print("[ERROR] Frame capture failed.")
            stopEvent.set()
            continue

        counter += 1
        annotatedFront = frameFront.copy()
        annotatedSide = frameSide.copy()

        # --- labda detektálás ---
        detectionsFront = detectorFront.infer(frameFront)
        detectionsSide = detectorSide.infer(frameSide)

        positionFront, positionSide = None, None
        try:
            positionFront = getCenter(detectionsFront.xyxy[0])
            positionSide = getCenter(detectionsSide.xyxy[0])
        except Exception:
            pass

        # --- Kalman előrejelzés ---

        impactFront, impactSide = None, None
        mappedFront,mappedSide = None, None

        if positionFront and positionSide:

            impactFront, mappedFront = kfFront.predictImpact(positionFront, goalLineFront)
            impactSide, mappedSide = kfSide.predictImpact(positionSide, goalLineSide)


        # --- megjelenítés ---
        cv2.line(annotatedFront, goalLineFront[0], goalLineFront[1], (0, 255, 0), 3)
        cv2.line(annotatedSide, goalLineSide[0], goalLineSide[1], (0, 255, 0), 3)

        try:
            annotatedFront = boxAnnotator.annotate(annotatedFront, detectionsFront)
            annotatedSide = boxAnnotator.annotate(annotatedSide, detectionsSide)

            if positionFront:
                cv2.circle(annotatedFront, positionFront, 8, (0, 255, 0), -1)
            if positionSide:
                cv2.circle(annotatedSide, positionSide, 8, (0, 255, 0), -1)

            if impactFront is not None:
                cv2.circle(annotatedFront, tuple(map(int, impactFront)), 10, (255, 0, 0), -1)
                cv2.putText(annotatedFront, f"Impact u={mappedFront:.3f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            if impactSide is not None:
                cv2.circle(annotatedSide, tuple(map(int, impactSide)), 10, (255, 0, 0), -1)
                cv2.putText(annotatedSide, f"Impact u={mappedSide:.3f}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        except Exception:
            pass

        # --- ablak megjelenítés ---
        frontWindow.showFrame(annotatedFront)
        sideWindow.showFrame(annotatedSide)
        goalWindow.showFrame(goalframe)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            stopEvent.set()

    cv2.destroyAllWindows()


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main(argsParser())
