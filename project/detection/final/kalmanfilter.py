# --------------------------------------------------------------------
# Implements multiple objects motion prediction using Kalman Filter
#
# Author: Sriram Emarose [sriram.emarose@gmail.com]
#
#
#
# --------------------------------------------------------------------
import cv2
import cv2 as cv
import numpy as np
import supervision as sv
import sys

from project.detection.final.calc import getCenter
from project.detection.types.ODModel import ColorDetectorModel

MAX_OBJECTS_TO_TRACK = 2


# Instantiate OCV kalman filter
class KalmanFilter:
    kf = cv.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        predicted = self.kf.predict()
        self.kf.correct(measured)
        return predicted


# Performs required image processing to get ball coordinated in the video
class ProcessImage:
    def __init__(self, detector):
        self.detector : ColorDetectorModel = detector
        self.width = 0
        self.height = 0

    def DetectObject(self):

        vid = cv.VideoCapture(2)
        vid.set(cv2.CAP_PROP_FPS,60)
        print(vid.get(cv.CAP_PROP_FPS))

        if (vid.isOpened() == False):
            print('Cannot open input video')
            return

        self.width = int(vid.get(3))
        self.height = int(vid.get(4))

        # Create Kalman Filter Object
        kfObjs = []
        predictedCoords = []
        for i in range(MAX_OBJECTS_TO_TRACK):
            kfObjs.append(KalmanFilter())
            predictedCoords.append(np.zeros((2, 1), np.float32))

        while (vid.isOpened()):
            rc, frame = vid.read()

            if rc:
                coords = self.DetectBall(frame)

                for i in range(len(coords)):
                    if i > MAX_OBJECTS_TO_TRACK:
                        break
                    cv2.circle(frame, (int(coords[i][0]), int(coords[i][1])), 3, (255, 0, 0), -1)
                    # print (' circle ',i, ' ', coords[i][0], ' ', coords[i][1])
                    predictedCoords[i] = kfObjs[i].Estimate(coords[i][0], coords[i][1])
                    frame = self.DrawPredictions(frame, coords[i][0], coords[i][1], predictedCoords[i])

                cv.imshow('Input', frame)

                if (cv.waitKey(300) & 0xFF == ord('q')):
                    break

            else:
                break

        vid.release()
        cv.destroyAllWindows()

    # Segment the green ball in a given frame
    def DetectBall(self, frame):
        results = self.detector.infer(frame)
        detections : sv.Detections = self.detector.getDetectionFromResult(results, (self.width,self.height))
        coords = []

        if detections is not None:
            try:
                center = getCenter(detections.xyxy[0])
                for (x, y) in center:
                    coords.append((x, y))
                return coords
            except:
                pass
        return coords

    def DrawPredictions(self, frame, ballX, ballY, predictedCoords):
        # Draw Actual coords from segmentation
        cv.circle(frame, (int(ballX), int(ballY)), 20, [0, 0, 255], 2, 8)
        cv.line(frame, (int(ballX), int(ballY + 20)), (int(ballX + 50), int(ballY + 20)), [100, 100, 255], 2, 8)
        cv.putText(frame, "Actual", (int(ballX + 50), int(ballY + 20)), cv.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

        # Draw Kalman Filter Predicted output
        cv.circle(frame, (int(predictedCoords[0]), int(predictedCoords[1])), 20, [0, 255, 255], 2, 8)
        cv.line(frame, (int(predictedCoords[0]) + 16, int(predictedCoords[1]) - 15),
                (int(predictedCoords[0]) + 50, int(predictedCoords[1]) - 30), [100, 10, 255], 2, 8)
        cv.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(predictedCoords[1] - 30)),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

        return frame


# Main Function
def main():
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    model = ColorDetectorModel(lower_green, upper_green)
    processImg = ProcessImage(detector=model)
    processImg.DetectObject()


if __name__ == "__main__":
    main()

print('Program Completed!')