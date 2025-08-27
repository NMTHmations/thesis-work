from enum import StrEnum


class ThreadNames(StrEnum):
    CAPTURE = "Capture"
    DETECTION = "Detection"
    VISUALIZER = "Visualizer"


class ModelTypes(StrEnum):
    YOLO = "Yolo"
    INFERENCE = "Inference"