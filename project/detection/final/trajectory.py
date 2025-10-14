from abc import abstractmethod
from typing import Optional

import cv2
import numpy as np


class Trajectory:
    def __init__(self, minimumNumberOfPoints):
        self.minimumNumberOfPoints = minimumNumberOfPoints

    @abstractmethod
    def predict(self, pointsX, pointsY):
        pass

    @abstractmethod
    def drawPrediction(self, frame, predictedPoints, drawBetween : Optional[tuple[tuple[int, int], tuple[int, int]]]):
        pass

    @abstractmethod
    def getMappedImpact(self, impactPoint, line: tuple[tuple[int, int], tuple[int, int]]):
        pass


class BallisicTrajectory_Singlecam(Trajectory):
    def getMappedImpact(self, impactPoint, line: tuple[tuple[int, int], tuple[int, int]]):
        pass

    #Polinomial regression y = Ax^2 + Bx + C
    def __init__(self, xAxis, minimumNumberOfPoints = 2):
        super().__init__(minimumNumberOfPoints)
        self.xAxis = xAxis


    def predict(self, pointsX, pointsY) -> list[tuple[int,int]]:
        predictedPoints = []

        if pointsX:
            A, B, C= np.polyfit(pointsX, pointsY,2)

            for x in self.xAxis:
                y = int(A * pow(x, 2) + B * 2 + C)
                predictedPoints.append((x,y))

        return predictedPoints

    def drawPrediction(self, frame, predictedPoints, drawBetween : Optional[tuple[tuple[int, int], tuple[int, int]]]):
        for point in predictedPoints:
            cv2.circle(frame, point,1, (255, 0, 255), cv2.FILLED)


class LinearTrajectory_Singlecam(Trajectory):
    def __init__(self, minimumNumberOfPoints = 2):
        super().__init__(minimumNumberOfPoints)


    def drawPrediction(self, frame, predictedPoints, drawBetween : Optional[tuple[tuple[int, int], tuple[int, int]]]):
        if not drawBetween:
            cv2.line(frame, predictedPoints[0], predictedPoints[1], (255, 0, 0), 2)

    def predict(self, pointsX, pointsY):
        x1, y1 = pointsX[-2], pointsY[-2]
        x2, y2 = (pointsX[-1], pointsY[-1])

        #Linear fun: y = mx+b

        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1

        return m,b

    def getMappedImpact(self, impactPoint, line: tuple[tuple[int, int], tuple[int, int]]):
        """
        Mapeli az impact_point x-koordinátáját 0-1 közé a line_start->line_end alapján,
        kiírja a frame-re és rajzol egy vízszintes indikátort a frame tetején.

        Parameters:
            impact_point (tuple): (x, y) pont
            line_start (tuple): a vonal kezdőpontja (x, y)
            line_end (tuple): a vonal végpontja (x, y)

        Returns:
            mapped_value (float): a pont mapelt értéke 0-1 között
            :param line:
            :param impactPoint:
        """
        line_start = line[0]
        line_end = line[1]

        x_min = line_start[0]
        x_max = line_end[0]
        x_ball = impactPoint[0]
        mapped_value = (x_ball - x_min) / (x_max - x_min)
        mapped_value = np.clip(mapped_value, 0, 1)

        return round(mapped_value, 4)
