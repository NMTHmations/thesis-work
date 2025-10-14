import os
import cv2
import numpy as np
from project.stereo_setup import paramsDir


class StereoCalibrator():
    def __init__(self, chessBoardSize, imagesLeft, imagesRight, square_size=1.0):
        self.chessBoardSize = chessBoardSize
        self.imagesLeft = imagesLeft
        self.imagesRight = imagesRight
        self.square_size = square_size  # a négyzet valós mérete (pl. m-ben vagy mm-ben)

    def calibrate(self):

        endCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        img_tmpL = cv2.imread(self.imagesLeft[0])
        img_tmpR = cv2.imread(self.imagesRight[0])
        grayScaleImgSize = self._getGrayscale(img_tmpL).shape[::-1]

        imgHeightLeft, imgWidthLeft, _ = img_tmpL.shape
        imgHeightRight, imgWidthRight, _ = img_tmpR.shape

        print("Calibrating...")

        objPoints, imgPointsLeft, imgPointsRight = self.collectPointsStereo(self.imagesLeft, self.imagesRight,
                                                                            endCriteria)

        _, cameraMatrixL, distortionCoefficientsL, _, _ = cv2.calibrateCamera(objPoints, imgPointsLeft,
                                                                              grayScaleImgSize, None, None)
        _, cameraMatrixR, distortionCoefficientsR, _, _ = cv2.calibrateCamera(objPoints, imgPointsRight,
                                                                              grayScaleImgSize, None, None)

        # Stereo kalibráció
        flags = cv2.CALIB_FIX_INTRINSIC
        retStereo, cameraMatrixL, distortionCoefficientsL, cameraMatrixR, distortionCoefficientsR, \
        R, T, E, F = cv2.stereoCalibrate(
            objPoints, imgPointsLeft, imgPointsRight,
            cameraMatrixL, distortionCoefficientsL,
            cameraMatrixR, distortionCoefficientsR,
            (imgWidthLeft, imgHeightLeft),
            criteria=endCriteria,
            flags=flags
        )

        print("Stereo RMS reprojection error: {:.3f}".format(retStereo))

        # Stereo rectifikáció
        rectifyScale = 1
        R1, R2, P1, P2, Q, roiL, roiR = cv2.stereoRectify(
            cameraMatrixL, distortionCoefficientsL,
            cameraMatrixR, distortionCoefficientsR,
            (imgWidthLeft, imgHeightLeft), R, T,
            alpha=rectifyScale
        )

        # Undistort/rectify mapek
        stereoMapLeft = cv2.initUndistortRectifyMap(
            cameraMatrixL, distortionCoefficientsL, R1, P1,
            (imgWidthLeft, imgHeightLeft), cv2.CV_16SC2
        )
        stereoMapRight = cv2.initUndistortRectifyMap(
            cameraMatrixR, distortionCoefficientsR, R2, P2,
            (imgWidthRight, imgHeightRight), cv2.CV_16SC2
        )

        print("Stereo calibration done ✅")

        # Elmentjük az összes fontos paramétert
        self.saveParams(
            cameraMatrixL, distortionCoefficientsL,
            cameraMatrixR, distortionCoefficientsR,
            R, T, E, F, R1, R2, P1, P2, Q,
            stereoMapLeft, stereoMapRight
        )

        self.saveParams(cameraMatrixL, distortionCoefficientsL,
                        cameraMatrixR, distortionCoefficientsR,
                        R, T, E, F, R1, R2, P1, P2, Q,
                        stereoMapLeft, stereoMapRight)

    def collectPointsStereo(self, imagesLeft, imagesRight, criteria):
        objPoints, imgPointsLeft, imgPointsRight = [], [], []

        objp = np.zeros((self.chessBoardSize[0] * self.chessBoardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessBoardSize[0], 0:self.chessBoardSize[1]].T.reshape(-1, 2)

        for imgL, imgR in zip(imagesLeft, imagesRight):
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

        cvFile.write("R1", R1)
        cvFile.write("R2", R2)
        cvFile.write("P1", P1)
        cvFile.write("P2", P2)
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
