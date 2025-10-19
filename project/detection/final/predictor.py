from abc import abstractmethod

import cv2
import numpy as np

class Predictor:

    @abstractmethod
    def predictImpact(self, center, points):
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

    def predictImpact(self, center, points):
        pred, post = self._predictAndCorrect(center=center)

        impactPoint = self._computeImpactOnLine(points)


    def _computeImpactOnLine(self, points):
        if not points:
            return None,None

        st = self.filter.statePost if self.filter.statePost else self.filter.statePre

        state = np.array([float(st[0]), float(st[1]), float(st[2]), float(st[3])])
        p = state[0:2]
        v = state[2:4]

        t, u = self._line_intersection_parametric(p, v, points)
        impact_pt = None
        if t is not None and t >= 0:
            # compute impact point from prediction
            impact_pt = p + v * t


        return impact_pt

    def _predictAndCorrect(self, center):
        prediction = self.filter.predict()
        if center is not []:
            measurements = np.array([[np.float32(center[0])], [np.float32(center[1])]])
            self.filter.correct(measurements)

            postState = self.filter.statePost

        else:
            postState = prediction

        return prediction, postState

    def _line_intersection_parametric(self, p, v, points):
        """
        Solve p + v*t = a + u*(b-a) for t and u.
        Returns (t, u) or (None, None) if degenerate.
        """

        p1,p2 = points[0], points[1]

        # Solve 2x2: v * t - (b-a) * u = a - p
        A = np.column_stack((v, -(p2 - p1)))  # 2x2
        rhs = p1 - p
        if np.linalg.matrix_rank(A) < 2:
            return None, None
        sol = np.linalg.solve(A, rhs)
        t, u = float(sol[0]), float(sol[1])
        return t, u

    def _mapPrediction(self, points, impactPoint):
        """Return u in [0,1] for projection of pt onto segment a->b"""

        p1, p2 = points[0], points[1]

        diff = p2 - p1
        denom = (diff @ diff)
        if denom == 0:
            return 0.0
        u = float((impactPoint - p1) @ diff / denom)
        return np.clip(u, 0.0, 1.0)

