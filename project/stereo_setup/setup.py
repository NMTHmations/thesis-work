import argparse
import glob
import os.path

from ImageCapturer import ImageCapturer
from StereoCalibrator import StereoCalibrator
from __init__ import *
def args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--left", type=int, default=0, help="camera number")
    argument_parser.add_argument("--right", type=int, default=1, help="camera number")
    return argument_parser.parse_args()
if __name__ == '__main__':

    args = args()
    cam1 = args.left
    cam2 = args.right

    ImageCapturer.clear()
    ImageCapturer.captureCalibrationImages(leftCamera=cam1, rightCamera=cam2)

    imagesL = glob.glob(os.path.join(mainDir, subDirs[0], "*"))
    imagesR = glob.glob(os.path.join(mainDir, subDirs[1], "*"))

    sc = StereoCalibrator((3,7),imagesL,imagesR)

    sc.calibrate()


