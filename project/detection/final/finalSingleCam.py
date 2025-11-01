#!/usr/bin/env python3
from collections import deque

import cv2
import numpy as np
import json
import supervision as sv
from argparse import ArgumentParser

from project.detection.final.predictor import KFPredictor
from project.detection.final.visualizeUtils import GoalVisualizer
from project.detection.types.Camera import Camera
from project.detection.types.ODModel import ColorDetectorModel
from project.detection.types.Window import Window

def argsParser():
    parser = ArgumentParser()
    parser.add_argument("--camera", type=str, default="/dev/video3", help="Camera id")
    parser.add_argument("--fps", type=float, default=60.0, help="Kamera FPS")
    parser.add_argument("--width", type=int, default=1280, help="Width of frames record")
    parser.add_argument("--height", type=int, default=720, help="Height of frames record")
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
    camera = Camera(args.camera, args.width, args.height, args.fps)

    dt = 1.0 / args.fps if args.fps > 0.0 else 1.0 / float(camera.camera.get(cv2.CAP_PROP_FPS))

    # --- megjelenítő ablakok ---
    window = Window("Front Preview", 896, 504)
    #goalWindow = Window("Goal Preview", 896, 504)

    # --- pontok betöltése JSON-ból ---
    points = loadPoints("points.json")


    goalLine = (points["GOAL1"], points["GOAL2"])

    print("Success")

    print(f"[INFO] Front goal line: {goalLine}")

    # --- detektor és KF inicializálás ---
    lowerHSV = (45, 167, 0)
    upperHSV = (89, 255, 255)

    detector = ColorDetectorModel(lowerHSV, upperHSV)

    boxAnnotator = sv.BoxAnnotator()
    traceAnnotator = sv.TraceAnnotator()
    tracker = sv.ByteTrack()

    predictor = KFPredictor(dt=dt)
    lastImpactPoint = ()
    lastMappedImpactPoint = -1.0


    success = True


    prevCenter = ()
    center = ()

    #visualizer = GoalVisualizer(fpath="goal.png")

    while success:

        success, frame = camera.capture()
        #goalFrame = visualizer.createFrame()


        if not success:
            print("Cannot capture frames")
            break

        # --- labda detektálás ---
        detections = detector.infer(frame)
        detections = tracker.update_with_detections(detections)

        prevCenter = center
        center = ()

        try:
            center = getCenter(detections.xyxy[0])

        except:
            pass

        # --- Kalman előrejelzés ---r

        impactPoint, mappedImpact = None, None

        if center != () and prevCenter != () and (
                abs(center[0] - prevCenter[0]) > 5.0 or abs(center[1] - prevCenter[1]) > 5.0):
            impactPoint, mappedImpact = predictor.predictImpact(center, goalLine)
            if impactPoint is not None:
                lastImpactPoint = tuple(map(int, impactPoint))
                lastMappedImpactPoint = float(mappedImpact)

                #print(f"[CONTROL] ImpactPoint sent: {lastImpactPoint}, u={mappedImpact:.3f}")

        annotated = frame.copy()
        #annotatedGoalFrame = goalFrame.copy()

        try:

            cv2.line(annotated, goalLine[0], goalLine[1], (0, 255, 0), 3)

            if center:
                annotated = boxAnnotator.annotate(annotated, detections)
                annotated = traceAnnotator.annotate(annotated, detections)
                cv2.circle(annotated, center, 10, (0,255,0), -1)

            if len(lastImpactPoint) == 2:

                cv2.circle(annotated, lastImpactPoint, 10, (255,0,0), -1)
                cv2.putText(annotated, f"Impact u={mappedImpact:.3f}", (lastImpactPoint[0],lastImpactPoint[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.6, (255,0,0), 2)

            #annotatedGoalFrame = visualizer.annotateFrame(annotatedGoalFrame,0.1, lastMappedImpactPoint)

        except:
            pass

        # --- ablak megjelenítés ---
        window.showFrame(annotated)
        #goalWindow.showFrame(annotatedGoalFrame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break
        elif key == ord('r'):
            predictor.reset()
            lastImpactPoint = ()
            predictor.impactSent = False


    del window
    del camera
    #del goalWindow


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    main(argsParser())
