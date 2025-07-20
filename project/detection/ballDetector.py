from ultralytics import YOLO
import cv2
import numpy as np



#object detection model
model = YOLO("yolo11n.pt")


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
         the path should be given in the following format: "../testVid.fileFormat"
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

    #for testing purposes
    for process in processVideo("../ballpassing2.mp4"):

        cv2.imshow("live", process[0])
        cv2.imshow("gray", process[1])
        cv2.imshow("detect", np.array(process[2]))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


