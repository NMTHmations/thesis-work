from enum import StrEnum, IntEnum


class ThreadNames(StrEnum):
    CAPTURE = "Capture"
    DETECTION = "Detection"
    VISUALIZER = "Visualizer"


class ModelTypes(StrEnum):
    YOLO = "Yolo"
    INFERENCE = "Inference"
    COLOR = "Color"


class FrameSize(IntEnum):
    INPUTDIM = 640