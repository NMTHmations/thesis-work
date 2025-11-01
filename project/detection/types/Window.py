import cv2


class Window:
    def __init__(self, name: str, width: int, height: int):
        self.name = name
        self._create(width, height)

    def _create(self, w, h):
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, w, h)

    def showFrame(self, frame):
        cv2.imshow(self.name, frame)

    def __del__(self):
        cv2.destroyWindow(self.name)
