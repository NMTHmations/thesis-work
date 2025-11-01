import cv2
import numpy as np

class GoalVisualizer:
    def __init__(self, frameSize=(896, 504), fpath="goal.png"):
        self.frameWidth, self.frameHeight = frameSize
        self.fpath = fpath

        # Átméretezett goal kép paraméterei
        self.goalWidth = 448
        self.goalHeight = 252

        # Középre pozicionálás
        self.xOffset = (self.frameWidth - self.goalWidth) // 2
        self.yOffset = (self.frameHeight - self.goalHeight) // 2

        # Létrehozott frame
        self.frame = self.createFrame()

    def createFrame(self):
        """Fehér háttér + goal PNG + vonalak és sarokpontok"""
        goalFrame = np.ones((self.frameHeight, self.frameWidth, 3), dtype=np.uint8) * 100

        # PNG betöltés alfa csatornával
        goalPic = cv2.imread(self.fpath, cv2.IMREAD_UNCHANGED)  # BGR + alfa
        goalPic = cv2.resize(goalPic, (self.goalWidth, self.goalHeight))

        bgr_img = goalPic[:, :, :3]
        alpha = goalPic[:, :, 3] / 255.0 if goalPic.shape[2] == 4 else np.ones((self.goalHeight, self.goalWidth))

        # ROI
        roi = goalFrame[self.yOffset:self.yOffset + self.goalHeight,
              self.xOffset:self.xOffset + self.goalWidth]

        for c in range(3):
            roi[:, :, c] = (bgr_img[:, :, c] * alpha + roi[:, :, c] * (1 - alpha)).astype(np.uint8)

        goalFrame[self.yOffset:self.yOffset + self.goalHeight,
        self.xOffset:self.xOffset + self.goalWidth] = roi

        # Vonalak és sarkok
        cv2.line(goalFrame, (self.xOffset, self.yOffset - 20), (self.frameWidth - self.xOffset, self.yOffset - 20), (0, 0, 255), 2, 1)
        cv2.line(goalFrame, (self.xOffset - 20, self.yOffset), (self.xOffset - 20, self.frameHeight - self.yOffset), (0, 0, 255), 2, 1)

        # Sarokpontok és szövegek
        cv2.circle(goalFrame, (self.xOffset, self.yOffset - 20), 5, (255, 0, 0), -1)
        cv2.circle(goalFrame, (self.frameWidth - self.xOffset, self.yOffset - 20), 5, (255, 0, 0), -1)
        cv2.putText(goalFrame, "0.0", (self.xOffset - 20, self.yOffset - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(goalFrame, "1.0", (self.frameWidth - self.xOffset - 10, self.yOffset - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.circle(goalFrame, (self.xOffset - 20, self.yOffset), 5, (255, 0, 0), -1)
        cv2.circle(goalFrame, (self.xOffset - 20, self.frameHeight - self.yOffset), 5, (255, 0, 0), -1)
        cv2.putText(goalFrame, "1.0", (self.xOffset - 55, self.yOffset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(goalFrame, "0.0", (self.xOffset - 55, self.frameHeight - self.yOffset + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        return goalFrame

    def annotateFrame(self, frame, height: float, width: float):
        """
        Rajzol egy pontot a goal frame-re a 0-1 arányban megadott koordináták alapján.
        height_ratio: 0.0 (top) -> 1.0 (bottom)
        width_ratio: 0.0 (left) -> 1.0 (right)
        """

        if height < 0 or width < 0:
            return frame


        # Kiszámoljuk a pixel koordinátákat
        x = int(self.xOffset + width * self.goalWidth)
        # Y-t fordítjuk
        y = int(self.yOffset + self.goalHeight - height * self.goalHeight)

        cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)

        return frame
