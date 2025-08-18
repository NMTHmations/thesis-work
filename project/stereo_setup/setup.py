import glob
import os.path

from ImageCapturer import ImageCapturer
from StereoCalibrator import StereoCalibrator
from __init__ import *

if __name__ == '__main__':

    #ImageCapturer.clear()
    #ImageCapturer.captureCalibrationImages()

    imagesL = glob.glob(os.path.join(mainDir, subDirs[0], "*"))
    imagesR = glob.glob(os.path.join(mainDir, subDirs[1], "*"))

    sc = StereoCalibrator((7,7),imagesL,imagesR)

    stereoMapL, stereoMapR = sc.calibrate()

    sc.saveParams((stereoMapL,stereoMapR))

