import cv2
import numpy as np


class KFPredictor:
    """
    Optimalizált OpenCV Kalman Filter alapú 2D pozíció- és becsapódásbecslés.
    """

    def __init__(self, dt: float):
        self.dt = dt
        self.filter = self._createFilter(dt)
        self.initialized = False
        self.previousMeasurement = None

    # -----------------------------
    # Kalman-szűrő létrehozása 2D-re
    # -----------------------------
    def _createFilter(self, dt: float) -> cv2.KalmanFilter:
        kf = cv2.KalmanFilter(4, 2)  # state [x,y,vx,vy], measurement [x,y]

        # Transition matrix
        kf.transitionMatrix = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ], dtype=np.float32)

        # Measurement matrix
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)

        # Process noise covariance
        q_pos = 1e-2
        q_vel = 1e-2
        kf.processNoiseCov = np.diag([q_pos, q_pos, q_vel, q_vel]).astype(np.float32)

        # Measurement noise covariance
        r_meas = 1.0
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * r_meas

        # Posteriori error estimate
        kf.errorCovPost = np.eye(4, dtype=np.float32)

        return kf

    # -----------------------------
    # Predict + Correct lépés
    # -----------------------------
    def _predictAndCorrect(self, measurement):
        pred = self.filter.predict()

        if measurement is not None:
            meas = np.array([[np.float32(measurement[0])],
                             [np.float32(measurement[1])]])
            self.filter.correct(meas)

        # mindig az aktuális statePost-ot olvassuk ki
        st = self.filter.statePost if self.filter.statePost is not None else pred

        return np.array([st[0], st[1], st[2], st[3]], dtype=np.float32).flatten()

    # -----------------------------
    # Becsapódási pont számítása
    # -----------------------------
    def predictImpact(self, measurement, impactLinePts):
        if measurement is None or len(measurement) != 2:
            return None, None

        # Inicializálás az első mérésből
        if not self.initialized and self.previousMeasurement is not None:
            initialVelocity = (np.array(measurement) - np.array(self.previousMeasurement)) / self.dt

            self.filter.statePost = np.array([
                [measurement[0]], [measurement[1]],
                [initialVelocity[0]], [initialVelocity[1]]
            ], dtype=np.float32)

            self.initialized = True

        self.previousMeasurement = measurement

        # Kalman lépés
        state = self._predictAndCorrect(measurement)
        p, v = state[0:2], state[2:4]

        # Becsapódás számítása
        impactPt, u = self._calculateImpactOnLine(p, v, impactLinePts)
        if impactPt is not None:
            return impactPt, u

        return None, None

    # -----------------------------
    # Parametrikus metszésszámítás (stabil)
    # -----------------------------
    def _calculateImpactOnLine(self, p, v, goalLineParams):
        a = np.array(goalLineParams[0], dtype=np.float32)
        b = np.array(goalLineParams[1], dtype=np.float32)

        # Solve p + v*t = a + u*(b-a)
        A = np.column_stack((v, -(b - a)))
        rhs = a - p

        try:
            t, u = np.linalg.solve(A, rhs).astype(np.float32).flatten()
        except np.linalg.LinAlgError:
            return None, None

        # Csak előrefelé mozgó metszések
        if t < 0:
            return None, None

        impactPoint = p + v * t

        # clamp u [0,1] és megfelelő végpont
        if u < 0.0:
            impactPoint = a
            u = 0.0
        elif u > 1.0:
            impactPoint = b
            u = 1.0

        return impactPoint, float(u)

    def reset(self):
        self.filter = self._createFilter(self.dt)
        self.initialized = False
        self.previousMeasurement = None
        return self.filter

    def __str__(self):
        return "Optimized 2D Kalman Predictor"
