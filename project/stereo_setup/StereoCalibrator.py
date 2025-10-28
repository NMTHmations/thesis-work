import os
import cv2
import numpy as np
from project.stereo_setup import paramsDir


class StereoCalibrator():
    def __init__(self, chessBoardSize, imagesLeft, imagesRight, square_size=0.025):
        self.chessBoardSize = chessBoardSize
        self.imagesLeft = imagesLeft
        self.imagesRight = imagesRight
        self.squareSize = square_size  # a négyzet valós mérete (pl. m-ben vagy mm-ben)

    def calibrate(self):

        endCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        img_tmpL = cv2.imread(self.imagesLeft[0])
        img_tmpR = cv2.imread(self.imagesRight[0])
        grayScaleImgSize = self._getGrayscale(img_tmpL).shape[::-1]

        imgHeightLeft, imgWidthLeft, _ = img_tmpL.shape
        imgHeightRight, imgWidthRight, _ = img_tmpR.shape

        print("Calibrating...")

        objPoints, imgPointsLeft, imgPointsRight = self.collectStereoPoints(self.imagesLeft, self.imagesRight,
                                                                            endCriteria)

        successL, cameraMatrixL, distL, rvecsL, tvecsL = cv2.calibrateCamera(objPoints, imgPointsLeft, grayScaleImgSize, None, None)
        successR, cameraMatrixR, distR, rvecsR, tvecsR = cv2.calibrateCamera(objPoints, imgPointsRight, grayScaleImgSize, None, None)

        newCameraMatrixL, roi_L = cv2.getOptimalNewCameraMatrix(cameraMatrixL, distL, (imgWidthLeft, imgHeightLeft), 1, (imgWidthLeft, imgHeightLeft))
        newCameraMatrixR, roi_R = cv2.getOptimalNewCameraMatrix(cameraMatrixR, distR, (imgWidthRight, imgHeightRight), 1, (imgWidthRight, imgHeightRight))


        # Stereo kalibráció
        flags = cv2.CALIB_FIX_INTRINSIC
        retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, \
        R, T, E, F = cv2.stereoCalibrate(
            objPoints, imgPointsLeft, imgPointsRight,
            newCameraMatrixL, distL,
            newCameraMatrixR, distR,
            (imgWidthLeft, imgHeightLeft),
            criteria=endCriteria,
            flags=flags
        )

        print("Stereo RMS reprojection error: {:.3f}".format(retStereo))

        # Stereo rectifikáció
        rectifyScale = 1
        rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R = cv2.stereoRectify(
            newCameraMatrixL, distL,
            newCameraMatrixR, distR,
            (imgWidthLeft, imgHeightLeft), R, T,
            alpha=rectifyScale
        )

        # Undistort/rectify mapek
        stereoMapLeft = cv2.initUndistortRectifyMap(
            newCameraMatrixL, distL, rectL, projMatrixL,
            (imgWidthLeft, imgHeightLeft), cv2.CV_16SC2
        )
        stereoMapRight = cv2.initUndistortRectifyMap(
            newCameraMatrixR, distR, rectR, projMatrixR,
            (imgWidthRight, imgHeightRight), cv2.CV_16SC2
        )

        print("Stereo calibration done ✅")

        # Elmentjük az összes fontos paramétert
        self.saveParams(
            newCameraMatrixL, distL,
            newCameraMatrixR, distR,
            R, T, E, F, rectL, rectR, projMatrixL, projMatrixR, Q,
            stereoMapLeft, stereoMapRight
        )

    def collectStereoPoints(self, imagesLeft, imagesRight, criteria):
        objPoints, imgPointsLeft, imgPointsRight = [], [], []

        objp = np.zeros((self.chessBoardSize[0] * self.chessBoardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessBoardSize[0], 0:self.chessBoardSize[1]].T.reshape(-1, 2)
        objp *= self.squareSize

        for imgL, imgR, i in zip(imagesLeft, imagesRight, range(len(imagesLeft))):
            imgL = cv2.imread(imgL)
            imgR = cv2.imread(imgR)

            grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
            grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

            foundL, cornersL = cv2.findChessboardCorners(grayL, self.chessBoardSize, None)
            foundR, cornersR = cv2.findChessboardCorners(grayR, self.chessBoardSize, None)

            if foundL and foundR:
                objPoints.append(objp)

                cornersL = cv2.cornerSubPix(grayL, cornersL, (11, 11), (-1, -1), criteria)
                cornersR = cv2.cornerSubPix(grayR, cornersR, (11, 11), (-1, -1), criteria)

                imgPointsLeft.append(cornersL)
                imgPointsRight.append(cornersR)

                # Debug: kirajzolás
                cv2.drawChessboardCorners(imgL, self.chessBoardSize, cornersL, foundL)
                cv2.drawChessboardCorners(imgR, self.chessBoardSize, cornersR, foundR)
                cv2.imshow("Left", imgL)
                cv2.imshow("Right", imgR)
                cv2.waitKey(500)
            else:
                print(f"cannot find chessboard corners on pic{i}")

        cv2.destroyAllWindows()
        return objPoints, imgPointsLeft, imgPointsRight

    def saveParams(self, camL, distL, camR, distR,
                   R, T, E, F, R1, R2, P1, P2, Q,
                   stereoMapLeft, stereoMapRight,
                   path=os.path.join(paramsDir, 'calibration.xml')):

        print(f"Saving calibration parameters to {path}")
        cvFile = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)

        cvFile.write("cameraMatrixL", camL)
        cvFile.write("distCoeffsL", distL)
        cvFile.write("cameraMatrixR", camR)
        cvFile.write("distCoeffsR", distR)

        cvFile.write("R", R)
        cvFile.write("T", T)
        cvFile.write("E", E)
        cvFile.write("F", F)

        cvFile.write("rectLeft", R1)
        cvFile.write("rectRight", R2)
        cvFile.write("projectionLeft", P1)
        cvFile.write("projectionRight", P2)
        cvFile.write("Q", Q)

        # Stereo mapeket két részre kell bontani
        stL_x, stL_y = stereoMapLeft
        stR_x, stR_y = stereoMapRight

        cvFile.write("stereoMapL_x", stL_x)
        cvFile.write("stereoMapL_y", stL_y)
        cvFile.write("stereoMapR_x", stR_x)
        cvFile.write("stereoMapR_y", stR_y)

        cvFile.release()
        print("Calibration file saved")

    def _getGrayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
