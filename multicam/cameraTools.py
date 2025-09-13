import cv2
import sys
import os
import time

class CameraTools():
    def __init__(self):
        os.system("v4l2-ctl --list-devices --list-formats-ext")

"""
os.environ["QT_QPA_PLATFORM"] = "xcb"

cap = cv2.VideoCapture("/dev/video0")
#cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
if not cap.isOpened():
    print("Error happened!")
    exit(2)
while True:
    ret, frame = cap.read()
    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    #if not ret:
    #    print("Error with showing the image")
    frame = cv2.resize(frame, (640,480))
    cv2.imwrite("Test.jpg",frame)
    cv2.imshow("Test Image", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
"""
