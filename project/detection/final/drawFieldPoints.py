import argparse
import cv2
import numpy as np
import supervision as sv

from project.detection.final.calc import getCenter, point_on_line
from project.detection.final.fieldutils import FieldUtils
from project.detection.final.trajectory import LinearTrajectory_Singlecam, Trajectory, BallisicTrajectory_Singlecam
from project.detection.types.ODModel import ColorDetectorModel


def args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--cam1", default=2, type=int)
    argument_parser.add_argument("--cam2", default=1, type=int)
    return argument_parser.parse_args()




def get_points(fpath):
    import json
    with open(fpath, "r") as f:
        points = json.load(f)

    return points




def main(cam1, cam2):
    cap1 = cv2.VideoCapture(cam1)
    cap2 = cv2.VideoCapture(cam2)

    cap1.set(cv2.CAP_PROP_FPS, 60)
    cap2.set(cv2.CAP_PROP_FPS, 60)

    print(cap1.get(cv2.CAP_PROP_FPS))
    print(cap2.get(cv2.CAP_PROP_FPS))

    points_S = get_points("points_S.json")
    points_F = get_points("points_F.json")

    isAthaladt_FirstLine = False
    isAthaladt_SecondLine = False
    isEnoughPoints = False
    isPredicted = False
    isSegment = False
    pointItSegments = ()
    mapped_value = None
    pointOnLine = None


    center = None

    width = 640
    height = 480

    trajectory : Trajectory = LinearTrajectory_Singlecam(minimumNumberOfPoints=10)

    black = cv2.imread("mask.png")
    black = cv2.resize(black, (width, height))

    collectedCentersX = []
    collectedCentersY = []

    prediction = None

    # Zöld szín tartománya HSV-ben
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    model = ColorDetectorModel(lower_green, upper_green)

    #model = YOLO("../../models/best.engine")
    boxAnnotator = sv.BoxAnnotator()
    traceAnnotator = sv.TraceAnnotator()
    tracker = sv.ByteTrack()

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if not ret1 or not ret2:
            print("❌ Nem sikerült beolvasni a képet")
            break

        detections = None
        annotated = None

        try:
            results = model.infer(frame1)
            detections = model.getDetectionFromResult(results,(640,480))

            # --- Tracker frissítése ---

            # --- Kirajzolás ---
            annotated = frame1.copy()
            annotated = boxAnnotator.annotate(scene=annotated, detections=detections)



        except Exception as e:
            annotated = frame1.copy()

        FieldUtils.drawPoints(annotated,points_S)
        FieldUtils.drawPoints(frame2,points_F)
        FieldUtils.drawField_SIDE(annotated,points_S)
        FieldUtils.drawField_FRONT(frame2,points_F)

        if detections is not None and not isAthaladt_FirstLine:

            # a loopon belül
            try:
                center = getCenter(detections.xyxy[0])
                if center[0] > points_S["S_PB1"][0]:
                    isAthaladt_FirstLine = True
            except Exception as e:
                pass

        if detections is not None and not isAthaladt_SecondLine:

            try:
                center = getCenter(detections.xyxy[0])
                if center[0] > points_S["S_PT2"][0]:
                    isAthaladt_SecondLine = True
            except Exception as e:
                pass



        if isAthaladt_FirstLine:
            cv2.line(annotated, points_S["S_PB1"], points_S["S_PT1"], (0, 255, 0), 2)

            if collectedCentersX.__len__() > trajectory.minimumNumberOfPoints:
                isEnoughPoints = True

            if not isEnoughPoints:
                collectedCentersX.append(center[0])
                collectedCentersY.append(center[1])

        else:
            cv2.line(annotated, points_S["S_PB1"], points_S["S_PT1"], (255, 255, 0), 2)

        if collectedCentersX:
            for x,y in zip(collectedCentersX, collectedCentersY):
                cv2.circle(annotated, (int(x),int(y)), 1, (0,255,255), 2)

        if isEnoughPoints and not isPredicted:
            try:
                prediction = FieldUtils.line_through_image(collectedCentersX, collectedCentersY, width)
                #prediction = trajectory.predict(collectedCentersX,collectedCentersY)
            except Exception as e:
                print("Nem lehet meghatározni a pontokat" + str(e))
                isEnoughPoints = False

        else:
            isEnoughPoints = False

        try:
            isSegment, pointItSegments = FieldUtils.line_segment_intersection(prediction[0],prediction[1],points_S["S_GOALT"],points_S["S_GOALB"])
        except:
            pass

        if isSegment and pointItSegments != ():
            cv2.circle(annotated,pointItSegments, 5, (255, 255, 255), -1)
            if prediction is not None and prediction != []:
                # trajectory.drawPrediction(annotated, prediction, None)
                cv2.line(annotated, (collectedCentersX[0],collectedCentersY[0]), pointItSegments, (0, 255, 0), 2)

            if mapped_value is None:
                mapped_value = trajectory.getMappedImpact(impactPoint=pointItSegments,
                                                 line= (points_S["S_GOALT"], points_S["S_GOALB"]))
                pointOnLine = point_on_line((100,100),(width-100,100),mapped_value)
                print("Mapped value:" + str(mapped_value))
            else:
                cv2.line(annotated, (100,100), (width-100,100), (0, 255, 255), 2)
                cv2.circle(annotated, pointOnLine, 5, (255, 0, 255), 2)


        if not isAthaladt_SecondLine:
            cv2.line(annotated, points_S["S_PB2"], points_S["S_PT2"], (255, 255, 0), 2)
        else:

            cv2.line(annotated, points_S["S_PB2"], points_S["S_PT2"], (0, 255, 0), 2)

        cv2.imshow("Oldalso kamera (Image1)", annotated)
        cv2.imshow("Masik kamera (Image2)", frame2)



        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    args = args()
    main(args.cam1, args.cam2)
