import cv2
import glob, os

"""
Recommended to use like the following 

############################################

ImageCapturer.clear()
ImageCapturer.captureCalibrationImages()

############################################
"""


class ImageCapturer():

    """
    Deletes every image from the folders

    Should be used before every re-calibration processes
    """
    @staticmethod
    def clear():
        files = glob.glob('images/left/*.*')
        files.extend(glob.glob('images/right/*.*'))

        for file in files:
            os.remove(file)

        print("Removed all left/right images")

    """
        Creates a maximum of 10 pictures using both cameras for calibration purposes.
        
        The process controlled by keyboard:
            q -> End capturing
            s (1st time) -> Shows potential suitable frame
                s (2nd time) -> Saves shown frames  
            Other keys -> Continues without saving frames 
    
        #################################################################################
        
        Will only work if both cameras are connected and accessible.
        
        For best results setup the cameras in stereo (https://docs.opencv.org/4.x/stereo_depth.jpg)
            and make sure the cameras are in a fixed, stable position.
        
        Recommended capturing chessboard images from different angels.
    
    """
    @staticmethod
    def captureCalibrationImages():

        capRight = cv2.VideoCapture(0)
        capLeft = cv2.VideoCapture(1)

        if not capRight.isOpened():
            raise IOError("Couldn't open webcam1 (Right).")

        if not capLeft.isOpened():
            raise IOError("Couldn't open webcam2 (Left).")

        COUNT = 0

        while capRight.isOpened():
            successRight, frameRight = capRight.read()
            successLeft, frameLeft = capLeft.read()

            if not successLeft or not successRight:
                raise Exception("Failed to read frame.")

            cv2.imshow('left', frameLeft)
            cv2.imshow('right', frameRight)

            showKey = cv2.waitKey(1) & 0xFF
            if showKey == ord('q'):
                cv2.destroyAllWindows()
                break
            elif showKey == ord('s'):

                capturedRight =  frameRight
                capturedLeft = frameLeft

                cv2.namedWindow('capturedRightFrame', cv2.WINDOW_AUTOSIZE)
                cv2.namedWindow('capturedLeftFrame', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('capturedRightFrame', capturedRight)
                cv2.imshow('capturedLeftFrame', capturedLeft)

                saveKey = cv2.waitKey(0) & 0xFF
                if saveKey == ord('s'):
                    COUNT = COUNT + 1
                    cv2.imwrite(f'images/left/imageL{COUNT}.png', frameLeft)
                    cv2.imwrite(f'images/right/imageR{COUNT}.png', frameRight)

                cv2.destroyWindow('capturedRightFrame')
                cv2.destroyWindow('capturedLeftFrame')


            if COUNT == 10:
                cv2.destroyAllWindows()
                break

        capRight.release()
        capLeft.release()