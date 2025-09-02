from enum import StrEnum, IntEnum


class ThreadNames(StrEnum):
    CAPTURE = "Capture"
    DETECTION = "Detection"
    VISUALIZER = "Visualizer"


class ModelTypes(StrEnum):
    YOLO = "Yolo"
    INFERENCE = "Inference"


class FrameSize(IntEnum):
    INPUTDIM = 640