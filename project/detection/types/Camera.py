import cv2


class Camera:
    def __init__(self, camID: int | str, captureWidth: int, captureHeight: int, fps: float):
        self.camera = cv2.VideoCapture(camID)
        self.fps = fps
        self.frameWidth = captureWidth
        self.frameHeight = captureHeight
        self._setCamera()
        print(f"Camera{camID} initialized. {str(self)}")

    def _setCamera(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frameWidth)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frameHeight)
        self.camera.set(cv2.CAP_PROP_FPS, self.fps)

    def capture(self):
        success, frame = self.camera.read()
        return success, frame

    def __str__(self):
        return f"Frame size: {self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)}, FPS: {self.camera.get(cv2.CAP_PROP_FPS)}"

    def __del__(self):
        self.camera.release()

