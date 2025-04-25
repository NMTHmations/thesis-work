import supervision as sv
from inference import get_model
import ultralytics
import cv2

model = ultralytics.YOLO("yolo11n.pt") # load model
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (640,384)) # resize frame
    if not ret:
        print("Error: Could not read frame.")
        break
    results = model(frame)[0] # run inference
    tracker = sv.ByteTrack() # create tracker
    detections = sv.Detections.from_ultralytics(results) # get detections
    detections = detections[detections["class_name"] != "person"] # filter for person class
    detections = detections[detections.confidence > 0.35] # filter for confidence > 0.5
    detections = tracker.update_with_detections(detections) # update tracker
    label_annotator = sv.LabelAnnotator() # get annotations
    box_annotator = sv.BoxAnnotator() # get boxing
    trace_annotator = sv.TraceAnnotator() # create trace annotator
    labels = [f"{results.names[class_name]} {confidence:.2f}" for class_name, confidence in zip(detections.class_id, detections.confidence)]
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    frame = trace_annotator.annotate(scene=frame, detections=detections) # annotate frame with detections
    

    """
    labels = detections.class_id
    boxes = detections.xyxy
    # Display the detections on the frame
    # I left this commented out in the case of compability issues
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    """

    cv2.imshow("YOLOv8 Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
