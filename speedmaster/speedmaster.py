import supervision as sv
from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture('test1.mp4')
frames = []

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)


model = YOLO("yolo11n.pt")

while True:
    for frame in frames:
        frame = cv2.resize(frame, (640,384))
        cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()