import argparse

import cv2
import numpy as np
import supervision as sv

from project.detection.final.calc import getCenter
from project.detection.final.fieldutils import FieldUtils
from project.detection.final.predictor import KFPredictor_2D
from project.detection.types.ODModel import ColorDetectorModel


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=1, help="Kamera index")

    return p.parse_args()

def main(args):
    cameraIDX = args.cam

    capture = cv2.VideoCapture(cameraIDX)

    if not capture.isOpened():
        print("Cannot open camera", cameraIDX)
        return

    capture.set(cv2.CAP_PROP_FPS,60)

    fps = capture.get(cv2.CAP_PROP_FPS)
    frameWidth = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Recording in:{frameWidth}x{frameHeight} (16:9) {fps}fps")

    fieldPoints = FieldUtils.readPoints("points_F.json")


    lowerHSV = [40, 40, 40]
    upperHSV= [80, 255, 255]

    detector = ColorDetectorModel(lowerHSV, upperHSV)
    boxAnnotator = sv.BoxAnnotator()
    labelAnnotator = sv.LabelAnnotator()


    dt = 1.0 / fps

    predictor = KFPredictor_2D(dt)

    hasCrossedLine = False



    while capture.isOpened():
        success, frame = capture.read()

        if not success:
            raise Exception("Failed to read frame")

        detections = sv.Detections.empty()
        try:
            results = detector.infer(frame)
            detections = detector.getDetectionFromResult(results, None)
        except Exception as e:
            print("Failed to detect points")

        centerPoint = []
        if not detections.is_empty():
            centerPoint = getCenter(detections.xyxy[0])

        prediction,postX,postY = None, None,None
        if hasCrossedLine:
            prediction, postX, postY = predictor.predictImpact(centerPoint)


        FieldUtils.drawField_FRONT(frame, fieldPoints)

        try:
            cv2.circle(frame, (int(postX), int(postY)), 6, (0,0,255), -1)
            cv2.putText(frame, f"pred: {int(postX)},{int(postY)}", (int(postX) + 10, int(postY)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (0, 0, 255), 1)
        except Exception as e:
            print("Failed to draw points")

        stateVector = predictor.filter.statePost


        p = np.array(stateVector[0:2],dtype=np.float32)
        v = np.array(stateVector[2:4],dtype=np.float32)

        impactX, impactY = None, None


if __name__ == '__main__':
    main(parser())