from ultralytics import YOLO

model = YOLO("../models/yolo11m.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolo11n.engine'
