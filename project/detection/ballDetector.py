from ultralytics import YOLO
import cv2
import numpy as np



#object detection model
"""
    Models are and should be in the project/models folder 
"""
model = YOLO("../models/yolo11l.engine")
source = "../sources/vid/real2.mp4"

#source = None


def processSingleFrame(frame):
    """
        Processes a single frame:
        - Converts it to grayscale
        - Runs YOLO to detect sports balls (class ID 32)
        - Returns both grayscale and detection overlay frames
    """

    # convert to grayscale
    graysclafeFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #class 32 -> sports ball
    result = model(frame, stream=True, classes=32)

    # end result
    results = list(result)

    #If any object were detected, draw it on the frame
    if results:
        detectedObjects = results[0].plot()
    else:
        detectedObjects = frame

    return graysclafeFrame, detectedObjects

def processVideo(path: str = None):
    """
        - Captures video from webcam and processes each frame.
        - Yields original, grayscale, and detection frames.

        In case testing with video file,
         the path should be given in the following format: "project/sources/vid/testVid.fileFormat"
    """



    if path:
        capture = cv2.VideoCapture(path)
    else:
        capture = cv2.VideoCapture(0)



    if not capture.isOpened():
        raise IOError("Cannot capture/open video.")

    try:
        while True:
            #Read frames
            ret, frame = capture.read()
            if not ret:
                break

            #frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)

            graysclafeFrame, detectedObjects = processSingleFrame(frame)

            yield frame, graysclafeFrame, detectedObjects

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        capture.release()


if __name__ == '__main__':

    """
        Main loop for running the video processing.
        - Displays the original, grayscale, and detection output.
    """
    windowsSize = (1280, 720)

    cv2.namedWindow("live", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("live", windowsSize[0], windowsSize[1])

    cv2.namedWindow("gray", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("gray", windowsSize[0], windowsSize[1])

    cv2.namedWindow("detect", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("detect", windowsSize[0], windowsSize[1])

    #for testing purposes
    for process in processVideo(source):

        rawFrame, grayscaleFrame, detectedFrame = process[0], process[1], np.array(process[2])



        #If you came across with the problem that the video is upside down include the following:
        
        """
        rawFrame = cv2.rotate(rawFrame, cv2.ROTATE_180)
        grayscaleFrame = cv2.rotate(grayscaleFrame, cv2.ROTATE_180)
        detectedFrame = cv2.rotate(np.array(detectedFrame), cv2.ROTATE_180)
        """


        cv2.imshow("live", rawFrame)
        cv2.imshow("gray", grayscaleFrame)
        cv2.imshow("detect", detectedFrame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


