import numpy as np


def interpolate(p1, p2, t):
    """Interpolálás két pont között, t=0->p1, t=1->p2"""
    x = (1 - t) * p1[0] + t * p2[0]
    y = (1 - t) * p1[1] + t * p2[1]
    return (int(round(x)), int(round(y)))


def point_on_line(p1, p2, value):
    """
    Megadja a pontot a p1-p2 vonalon a 0-1 közötti value alapján.

    Parameters:
        p1 (tuple): (x1, y1) kezdőpont
        p2 (tuple): (x2, y2) végpont
        value (float): 0..1 közötti érték, 0=p1, 1=p2

    Returns:
        tuple: (x, y) a vonalon lévő pont
    """
    value = np.clip(value, 0, 1)
    x = int(round(p1[0] + value * (p2[0] - p1[0])))
    y = int(round(p1[1] + value * (p2[1] - p1[1])))
    return x, y


def getCenter(xyxy):
    x1, y1, x2, y2 = xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


