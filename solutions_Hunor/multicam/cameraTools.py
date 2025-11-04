import cv2
import sys
import os
import time

class CameraTools():
    def __init__(self):
        os.system("v4l2-ctl --list-devices --list-formats-ext")