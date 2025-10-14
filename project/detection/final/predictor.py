from abc import abstractmethod

import cv2
import numpy as np

class Predictor:

    @abstractmethod
    def predictImpact(self, center) -> np.ndarray:
        pass


class KFPredictor_2D(Predictor):
    """
    Opencv Kalman Filter
    https://docs.opencv.org/3.4/dd/d6a/classcv_1_1KalmanFilter.html#af19be9c0630d0f658bdbaea409a35cda
    """
    def __init__(self,dt):
        self.filter = self._createFilter(dt)

    def _createFilter(self, dt : int) -> cv2.KalmanFilter:
        obj = cv2.KalmanFilter(4, 2)
        obj.transitionMatrix = np.array(
            [
                [1, 0, dt, 0],
                [0, 1, 0, dt],
                [0, 0, 1, 0 ],
                [0, 0, 0, 1 ],
            ], dtype=np.float32
        )

        obj.measurementMatrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0]
            ], dtype=np.float32
        )

        #Process noise covariance (Q)
        qPosition = 1e-2 #0.01
        qVelocity = 1e-2 #0.01

        obj.processNoiseCov = np.diag(
            [qPosition,qPosition,qVelocity,qVelocity]
        ).astype(np.float32)

        #Measurement noise (R)
        rMeasurement = 1.0

        obj.measurementNoiseCov = np.eye(2, dtype=np.float32) * rMeasurement

        #Posterior error estimate (P)
        obj.errorCovPost = np.eye(4, dtype=np.float32)

        return obj

    def predictImpact(self, center) -> np.ndarray:
        pass



    def util(self, center):

        prediction = self.filter.predict()
        if center is not []:
            measurements = np.array([[np.float32(center[0])], [np.float32(center[1])]])
            self.filter.correct(measurements)

            postState = self.filter.statePost

            postX,postY = float(postState[0]), float(postState[1])

        else:

            postX, postY = float(prediction[0]), float(prediction[1])

        return prediction, postX, postY