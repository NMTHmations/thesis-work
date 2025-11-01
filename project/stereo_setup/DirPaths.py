from enum import StrEnum

class DirPaths(StrEnum):
    MAINDIR = "images"
    LEFTDIR = "left"
    RIGHTDIR = "right"
    PARAMETERSDIR = "parameters"


"""

if __name__ == '__main__':
    img = cv2.imread("../sources/img/ball.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    width, height, channel = img.shape

    print(width, height, channel)
"""
