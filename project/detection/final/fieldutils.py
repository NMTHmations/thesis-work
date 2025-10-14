import cv2
import numpy as np


class FieldUtils:

    @staticmethod
    def drawPoints(frame,points):
        for name, (x, y) in points.items():
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, name, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

    @staticmethod
    def drawField_SIDE(frame, points):

        cv2.line(frame, points["S_TOPL"], points["S_TOPR"], (255, 0, 0), 2)
        cv2.line(frame, points["S_TOPR"], points["S_BOTR"], (255, 0, 0), 2)
        cv2.line(frame, points["S_BOTR"], points["S_BOTL"], (255, 0, 0), 2)
        cv2.line(frame, points["S_BOTL"], points["S_TOPL"], (255, 0, 0), 2)
        cv2.line(frame, points["S_GOALT"], points["S_GOALB"], (0, 255, 255), 2)

    @staticmethod
    def drawField_FRONT(frame, points):
        cv2.line(frame, points["F_TOPL"], points["F_TOPR"], (255, 0, 0), 2)
        cv2.line(frame, points["F_TOPR"], points["F_BOTR"], (255, 0, 0), 2)
        cv2.line(frame, points["F_BOTR"], points["F_BOTL"], (255, 0, 0), 2)
        cv2.line(frame, points["F_BOTL"], points["F_TOPL"], (255, 0, 0), 2)

    @staticmethod
    def readPoints(fpath):
        import json
        with open(fpath, "r") as f:
            points = json.load(f)

        return points

    @staticmethod
    def line_through_image(points, width, height):
        """
        Két pont alapján meghatározza az egyenes képernyőn belüli szakaszát.
        p1, p2: (x, y) pontok
        width, height: kép méretek
        return: [(x1, y1), (x2, y2)] vagy üres lista ha nincs metszés
        """
        p1 = points[0]
        p2 = points[1]

        x1, y1 = p1
        x2, y2 = p2

        dx = x2 - x1
        dy = y2 - y1

        points = []

        # bal szélnél (x=0)
        if dx != 0:
            t = -x1 / dx
            y = int(y1 + t * dy)
            if 0 <= y <= height:
                points.append((0, y))

        # jobb szélnél (x=width-1)
        if dx != 0:
            t = (width - 1 - x1) / dx
            y = int(y1 + t * dy)
            if 0 <= y <= height:
                points.append((width - 1, y))

        # felső szélnél (y=0)
        if dy != 0:
            t = -y1 / dy
            x = int(x1 + t * dx)
            if 0 <= x <= width:
                points.append((x, 0))

        # alsó szélnél (y=height-1)
        if dy != 0:
            t = (height - 1 - y1) / dy
            x = int(x1 + t * dx)
            if 0 <= x <= width:
                points.append((x, height - 1))

        # ha megvan két metszéspont
        if len(points) >= 2:
            return points[:2]
        else:
            return []
    @staticmethod
    def line_segment_intersection(A, B, C, D, eps=1e-9):
        """
        Meghatározza, hogy az AB egyenes és a CD szakasz találkozik-e.

        A, B, C, D: (x,y) pontok
        eps: numerikus tolerancia

        Visszatér:
          - (True, (x,y)) ha van metszéspont
          - (False, None) ha nincs
        """
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        C = np.array(C, dtype=float)
        D = np.array(D, dtype=float)

        AB = B - A
        CD = D - C

        denom = AB[0] * CD[1] - AB[1] * CD[0]

        if abs(denom) < eps:
            # Párhuzamosak vagy egybeesők
            return False, None

        # Paraméterek kiszámítása
        AC = C - A
        t = (AC[0] * CD[1] - AC[1] * CD[0]) / denom
        u = (AC[0] * AB[1] - AC[1] * AB[0]) / denom

        # Metszéspont az egyenesen mindig létezik, de u-t korlátozzuk [0,1]-re
        if 0 <= u <= 1:
            intersection = A + t * AB
            return True, (int(round(intersection[0])), int(round(intersection[1])))
        else:
            return False, None

    @staticmethod
    def shrink_segment(P1, P2, k=0.5):
        """
        P1, P2: tuple (x,y) - az eredeti szakasz végpontjai
        k: float (0<k<1) - arány, mekkora legyen az új szakasz
        """
        P1 = np.array(P1, dtype=float)
        P2 = np.array(P2, dtype=float)
        M = (P1 + P2) / 2  # középpont

        P1_new = M + k * (P1 - M)
        P2_new = M + k * (P2 - M)

        return tuple(P1_new), tuple(P2_new)

