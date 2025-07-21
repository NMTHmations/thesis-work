import torch
import supervision as sv
import cv2
import numpy as np
import albumentations as abm

class PerspectiveTransform:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        self.source = source.astype(np.float32)
        self.target = target.astype(np.float32)
        self.matrix = cv2.getPerspectiveTransform(self.source, self.target) # calculate the perspective transformation matrix

    # TODO: COORDINATE TRACKING
    
    def apply(self, frame):
        return cv2.warpPerspective(frame, self.matrix, (frame.shape[1], frame.shape[0])) # apply the perspective transformation to the frame

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
cap = cv2.VideoCapture("test1.mov")
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Task to do: Determine the source points
source = np.array([
    [50, 180],
    [260, 180],
    [230, 384],
    [0, 384]
    ]) # source points

# Define the target points for the perspective transformation
# These points should be in the same order as the source points

target = np.array([
    [0,0],
    [40,0],
    [40,250],
    [0,250]
    ]) # target points

transformer = PerspectiveTransform(source,target)

tracker = sv.ByteTrack() # create tracker

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# TODO: speed measurement
def detection_frame(frame:tuple):
    # albumentations usage
    blur_image = abm.OneOf([
        abm.MotionBlur(p=0.5),
        abm.GaussianBlur(p=0.5)
    ], p=0.4)
    transformed = blur_image(image=frame)
    img = transformed["image"]
    results = model(frame) # run inference
    res_tr = model(img)
    detections = sv.Detections.from_yolov5(results) # get detections
    detections1 = sv.Detections.from_yolov5(res_tr)
    detections = detections[detections.confidence > 0.35] # filter for confidence > 0.5
    detections = tracker.update_with_detections(detections) # update tracker
    detections = tracker.update_with_detections(detections1) # update tracker
    label_annotator = sv.LabelAnnotator() # get annotations
    box_annotator = sv.BoxAnnotator() # get boxing
    trace_annotator = sv.TraceAnnotator() # create trace annotator
    labels = [f"{results.names[class_name]} {confidence:.2f}" for class_name, confidence in zip(detections.class_id, detections.confidence)]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = trace_annotator.annotate(scene=frame, detections=detections) # annotate frame with detections
    return frame

# Test the model with the detection of a ball - currently it detects as volleyball instead of tennis ball
# and the confidence is relatively low - 0.65
image = cv2.imread("test.jpg") # read image
image = detection_frame(image) # run detection
cv2.imshow("YOLOv5 Detection of image", image) # show image

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)
        continue
    frame = cv2.resize(frame, (640,384)) # resize frame
    new_frame = transformer.apply(frame)
    if not ret:
        print("Error: Could not read frame.")
        break
    frame = detection_frame(frame) # run detection
    for point in source:
        cv2.circle(frame, tuple(map(int, point)), 5, (0, 255, 0), -1)
    cv2.imshow("YOLOv5 Detection", frame)
    cv2.imshow("Perspective transform",new_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
