import cv2
from ballDetector import *
import numpy as np

model_path = "../models/yolo11n.engine"
#model_path = "experiment-sxxxi/1"
source = "../sources/vid/real4.mp4"
#source = None

def main():
    """
        Main loop for running the video processing.
        - Displays the original, grayscale, and detection output.
    """

    balldetector = BallDetector_YOLO_CV2(model_path)
    #balldetector = BallDetector_SV_RF(model_path)

    windowSize = (1280, 720)

    #tracker = cv2.TrackerMOUSSE_create

    cv2.namedWindow("live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("live", windowSize[0], windowSize[1])

    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gray", windowSize[0], windowSize[1])

    cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("detect", windowSize[0], windowSize[1])

    # for testing purposes
    for process in balldetector.processVideo(source):

        rawFrame, grayscaleFrame, detectedFrame = process[0], process[1], np.array(process[2])

        # If you came across with the problem that the video is upside down include the following:

        """
        rawFrame = cv2.rotate(rawFrame, cv2.ROTATE_180)
        grayscaleFrame = cv2.rotate(grayscaleFrame, cv2.ROTATE_180)
        detectedFrame = cv2.rotate(np.array(detectedFrame), cv2.ROTATE_180)
        """

        cv2.imshow("live", rawFrame)
        #cv2.imshow("gray", grayscaleFrame)
        cv2.imshow("detect", detectedFrame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()