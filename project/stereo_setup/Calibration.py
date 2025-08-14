import cv2

from . import *
import glob
class StereoVision():
    def __init__(self):
        pass

    def calibrate(self):

        endCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        objPoints = []
        imgPoints = []

        chessBoardSize = (9,6)

        images = []

        try:
            images.extend(glob(os.path.join(mainDir, subDirs[0], "*.*")))
        except Exception as e:
            print(e)

        for fname in images:
            img = cv2.imread(fname)
            grayScale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(grayScale, chessBoardSize, None)

            if ret:
                objPoints.append(corners)

                subcorners = cv2.cornerSubPix(grayScale, corners, (11, 11), (-1, -1), endCriteria)
                imgPoints.append(subcorners)

                cv2.drawChessboardCorners(img, chessBoardSize, subcorners, ret)
                cv2.imshow("img", img)
                cv2.waitKey(0)

        cv2.destroyAllWindows()