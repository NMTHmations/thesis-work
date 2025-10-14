
import numpy as np
import cv2

from project.detection.types.ODModel import ColorDetectorModel, YOLOModel
import supervision as sv



def hughcircle(frame):
    # Apply Hough transform to greyscale image
    circles = cv2.HoughCircles(frame, cv2.HOUGH_GRADIENT, 1, minDist=480 / 16,
                               param1=100, param2=30, minRadius=40, maxRadius=50)
    circles = np.uint16(np.around(circles))

    return circles

image = cv2.imread("../../sources/img/frame-2.png")
image = cv2.resize(image, (480, 480))

boxannotator = sv.BoxAnnotator()
labelannotator = sv.LabelAnnotator(text_position=sv.Position.TOP_CENTER)

model2 = YOLOModel("../../models/yolo11l.engine",device=0,inferenceImgSize=480)
model3 = YOLOModel("../../models/best.engine",device=0,inferenceImgSize=480)

imageYOLO = image.copy()
imageYOLO2 = image.copy()


results2 = model2.infer(imageYOLO)
detections2 = model2.getDetectionFromResult(results2[0])
imageYOLO = boxannotator.annotate(imageYOLO, detections2)
labels2 = [f"Sports ball: {s:.2f}" for s in detections2.confidence]
imageYOLO = labelannotator.annotate(imageYOLO, detections2, labels=labels2)

results3 = model3.infer(imageYOLO2)
detections3 = model3.getDetectionFromResult(results3[0])
imageYOLO2 = boxannotator.annotate(imageYOLO2, detections3)
labels3 = [f"Ball: {s:.2f}" for s in detections3.confidence]
imageYOLO2 = labelannotator.annotate(imageYOLO2, detections3, labels=labels3)

imageColor = image.copy()

lower = np.array([40, 40, 40])
upper = np.array([80, 255, 255])
model = ColorDetectorModel(lower,upper)

results = model.infer(imageColor)
detections = model.getDetectionFromResult(results, (480,480))
imageColor = boxannotator.annotate(imageColor, detections)


imgHughcircle = cv2.medianBlur(image, 3)

# Convert to greyscale
imgHughcircle_gray = cv2.cvtColor(imgHughcircle, cv2.COLOR_BGR2GRAY)

circles = hughcircle(imgHughcircle_gray)

# Draw the circles
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(imgHughcircle,(i[0],i[1]),i[2],(0,0,255),4)
    # draw the center of the circle
    cv2.circle(imgHughcircle,(i[0],i[1]),2,(255,0,255),3)
cv2.imshow('detected circles',imgHughcircle)
cv2.imshow('detected circles2',imageColor)
cv2.imshow('detected circles3',imageYOLO)
cv2.imshow('detected circles4',imageYOLO2)
cv2.waitKey(0)


cv2.imwrite("hughcircle.jpg", imgHughcircle)
cv2.imwrite('colorcircle.jpg',imageColor)
cv2.imwrite('yolocircle_yolo11l.jpg',imageYOLO)
cv2.imwrite('yolocircle_best.jpg',imageYOLO2)
cv2.destroyAllWindows()