from abc import abstractmethod

import cv2
import numpy as np


class KFPredictor:
    """
    Opencv Kalman Filter
    https://docs.opencv.org/3.4/dd/d6a/classcv_1_1KalmanFilter.html#af19be9c0630d0f658bdbaea409a35cda
    """
    def __init__(self,dimensions, dt):
        self.dimensions = dimensions
        self.dt = dt
        self.filter = self._createFilter(dimensions, dt)

    def _createFilter(self, dims : int, dt : float) -> cv2.KalmanFilter:

        # (position, velocity) e.g. dims = 2 -> [x,y,vx,vy]
        stateSize = 2 * dims

        measurementSize = dims

        obj = cv2.KalmanFilter(stateSize, measurementSize)

        obj.transitionMatrix = np.eye(stateSize, dtype=np.float32)
        for i in range(dims):
            obj.transitionMatrix[i, i + dims] = dt

        obj.measurementMatrix = np.zeros((measurementSize, stateSize), dtype=np.float32)
        for i in range(dims):
            obj.measurementMatrix[i, i] = 1.0


        #Process noise covariance (Q)
        qPosition = 1e-2 #0.01
        qVelocity = 1e-2 #0.01

        obj.processNoiseCov = np.eye(stateSize, dtype=np.float32)
        for i in range(dims):
            obj.processNoiseCov[i, i] = qPosition
            obj.processNoiseCov[i + dims, i + dims] = qVelocity


        obj.measurementNoiseCov = np.eye(measurementSize, dtype=np.float32)
        #Posterior error estimate (P)
        obj.errorCovPost = np.eye(stateSize, dtype=np.float32)

        return obj


    def _predictAndCorrect(self, measurement):
        prediction = self.filter.predict()
        if measurement is not [] or not () and not None:
            measurement = np.array([[np.float32(measurement[0])], [np.float32(measurement[1])]])
            self.filter.correct(measurement)

            postState = self.filter.statePost

        else:
            postState = prediction

        x,y,vx,vy = self.filter.statePost if self.filter.statePost is not None else prediction

        state = np.array([x,y,vx,vy]).astype(np.float32)

        return state

    def predictImpact(self, measurement : tuple[float, float], impactLinePts : tuple[tuple[int,int],tuple[int,int]]):
        """
        Updates filter, predicts trajectory, and finds intersection with impact line.

        Args:
            measurement: (x, y) tuple — latest detection
            impactLinePts: [(x1,y1), (x2,y2)] — two endpoints of the impact line

        Returns:
            dict with fields:
                - 'impactPt': np.ndarray or None (pixel coords)
                - 'mappedImpact': np.ndarray or None (mapped impact point on line)
        """

        if measurement == () or None:
            return None,None

        state = self._predictAndCorrect(measurement)
        p = state[0:2]
        v = state[2:4]

        # If the velocity is too small, no reliable intersection

        if abs(np.linalg.norm(v)) < 1.0:
            return None, None

        # Calculate intersection
        impactPt, mappedImpactPt = self._calculateImpactOnLine(p, v, impactLinePts)
        return impactPt, mappedImpactPt

    # -----------------------------
    # Helper: intersection
    # -----------------------------
    def _calculateImpactOnLine(self, p, v, impactLinePts):
        """
        Calculates intersection point of a parametric line p+v*t with a segment a->b.
        Returns (impact_point, u_param) or (None, None)
        """
        a = np.array(impactLinePts[0], dtype=float)
        b = np.array(impactLinePts[1], dtype=float)

        A = np.column_stack((v, -(b - a)))  # 2x2 system
        rhs = a - p

        if np.linalg.matrix_rank(A) < 2:
            return None, None

        sol = np.linalg.solve(A, rhs).flatten()
        t, u = float(sol[0]), float(sol[1])

        if t < 0:  # intersection is "behind" the current position
            return None, None

        impact_point = p + v * t
        return impact_point, u


    def reset(self):
        return self._createFilter(dims=self.dimensions, dt=self.dt)

    def __str__(self):
        return f"Kalman Filter in {self.dimensions} dimensions"