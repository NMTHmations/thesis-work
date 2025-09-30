import os

import cv2
import numpy as np

from project.stereo_setup import paramsDir


class StereoCalibrator():
    def __init__(self, chessBoardSize, imagesLeft, imagesRight):
        self.chessBoardSize = chessBoardSize
        self.imagesLeft = imagesLeft
        self.imagesRight = imagesRight

    def calibrate(self):

        ######INIT######
        endCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        img_tmpL = cv2.imread(self.imagesLeft[0])
        img_tmpR = cv2.imread(self.imagesRight[0])
        grayScaleImgSize = self._getGrayscale(img_tmpL).shape[::-1]

        imgHeightLeft, imgWidthLeft, channelsLeft = img_tmpL.shape
        imgHeightRight, imgWidthRight, channelsRight = img_tmpR.shape

        print("Calibrating...")

        objPointsLeft, imgPointsLeft, = self.collectPoints(self.imagesLeft, endCriteria)
        objPointsRight, imgPointsRight = self.collectPoints(self.imagesRight, endCriteria)

        reprojectionErrorL, cameraMatrixL, distortionCoefficientsL, rotationVectorL, transitionVectorL = (
            cv2.calibrateCamera(objPointsLeft,
                                imgPointsLeft,
                                grayScaleImgSize,
                                None,
                                None)
            )

        reprojectionErrorR, cameraMatrixR, distortionCoefficientsR, rotationVectorR, transitionVectorR = (
            cv2.calibrateCamera(objPointsRight,
                                imgPointsRight,
                                grayScaleImgSize,
                                None,
                                None)
        )

        optimalCameraMatrixL, roiL = cv2.getOptimalNewCameraMatrix(
            cameraMatrixL,
            distortionCoefficientsL,
            (imgWidthLeft, imgHeightLeft),
            1,
            (imgWidthLeft, imgHeightLeft)) #New image size

        optimalCameraMatrixR, roiR = cv2.getOptimalNewCameraMatrix(
            cameraMatrixR,
            distortionCoefficientsR,
            (imgWidthRight, imgHeightRight),
            1,
            (imgWidthRight, imgHeightRight)
        )

        (retStereo, stereoCameraMatrixL, stereodistortionL, stereoCameraMatrixR, stereodistortionR,
         stereoRotationVector, stereoTransitionVector, stereoEssentialMatrix, stereoFundamentalMatrix)  = (
            cv2.stereoCalibrate(objPointsLeft, imgPointsLeft, imgPointsRight, optimalCameraMatrixL,distortionCoefficientsL,
                                optimalCameraMatrixR, distortionCoefficientsR, (imgWidthLeft, imgHeightLeft),endCriteria, cv2.CALIB_FIX_INTRINSIC))


        rectifyScale = 1

        rectLeft, rectRight, projectionMatrixLeft, projectionMatrixRight, Q, roiL, roiR = (
            cv2.stereoRectify(optimalCameraMatrixL, stereodistortionL,optimalCameraMatrixR,stereodistortionR,
                              (imgWidthLeft, imgHeightLeft), stereoRotationVector, stereoTransitionVector, rectifyScale, (0,0)))

        stereoMapLeft = cv2.initUndistortRectifyMap(optimalCameraMatrixL, stereodistortionL,rectLeft,projectionMatrixLeft,(imgWidthLeft, imgHeightLeft),cv2.CV_16SC2)
        stereoMapRight = cv2.initUndistortRectifyMap(optimalCameraMatrixR,stereodistortionR,rectRight,projectionMatrixRight,(imgWidthRight, imgHeightRight),cv2.CV_16SC2)



        print("Stereo calibration done")
        print(f"Object Points: {objPointsLeft}")
        print("Reprojection error: {:.3f}".format(retStereo))
        print(f"Stereo rotation vector:\n{stereoRotationVector}")
        print(f"Stereo transition vector:\n{stereoTransitionVector}")
        print(f"StreoMapLeft:\n {stereoMapLeft}")
        print(f"StreoMapRight:\n {stereoMapRight}")

        return stereoMapLeft, stereoMapRight

        #self.saveParams(reprojectionError, cameraMatrix, distortionCoefficients, rotationVector, transitionVector)

        #print(f"Camera matrix:\n {cameraMatrix}")
        #print("reprojection error (pixels): {:.4f}".format(reprojectionError))


    def collectPoints(self, images, criteria):

        objPoints, imgPoints = [], []

        objp = np.zeros((self.chessBoardSize[0]*self.chessBoardSize[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessBoardSize[0], 0:self.chessBoardSize[1]].T.reshape(-1, 2)

        #########
            #if we know the distance of two img points we should multiply the objp with that number
        #########


        for fname in images:
            img = cv2.imread(fname)

            grayScale = self._getGrayscale(img)

            cornersFound, corners = cv2.findChessboardCorners(grayScale, self.chessBoardSize, None)

            if cornersFound:
                objPoints.append(objp)

                subCorners = cv2.cornerSubPix(grayScale, corners, (11, 11), (-1, -1), criteria)

                imgPoints.append(subCorners)

                cv2.drawChessboardCorners(img, self.chessBoardSize, subCorners, cornersFound)
                resized = cv2.resize(img, (1280,720))
                cv2.imshow("img", resized)
                cv2.waitKey(1000)

        cv2.destroyAllWindows()

        return objPoints, imgPoints

    #yet to implement
    def saveStereoMaps(self, params : tuple, path = os.path.join(paramsDir, 'calibration.xml')):
        print(f"Saving parameters to {path}")
        cvFile = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)


        stereomapLeft = params[0]
        stereomapRight = params[1]

        stL_x, stL_y = stereomapLeft
        stR_x, stR_y = stereomapRight

        cvFile.write("stereoMapL_x", stL_x)
        cvFile.write("stereoMapL_y", stL_y)
        cvFile.write("stereoMapR_x", stR_x)
        cvFile.write("stereoMapR_y", stR_y)

        cvFile.release()

    #yet to implement
    def removeDistortion(self):
        pass


    def _getGrayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)