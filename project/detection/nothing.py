"""
Realtime pipeline:
- CaptureThread -> SharedBuffer (deque + Condition)
- DetectionThread (batch_size, model.predict(batch)) -> display_queue
- DisplayThread -> imshow annotated frames

Requirements:
- ultralytics, supervision, opencv-python
- model_path: lehet .pt vagy más, de ellenőrizd, hogy a model.predict batch-et támogatja-e (PT általában ok)
"""

import threading
import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import cv2
from inference import get_model
from ultralytics import YOLO
import supervision as sv
import numpy as np
import traceback


@dataclass
class FrameItem:
    frame_id: int
    frame: any
    timestamp: float


class SharedBuffer:

    def __init__(self, maxlen: int = 512):
        #Deque for O(1) methods
        self.items: Deque[FrameItem] = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)

    def push(self, item: FrameItem):
        with self.lock:

            #A legrégebbi frameket eldobjuk
            if len(self.items) == self.items.maxlen:
                self.items.popleft()

            self.items.append(item)
            self.not_empty.notify()

    def pop_batch(self, batch_size: int, timeout: Optional[float] = None) -> List[FrameItem]:
        """
        Vár a timeout másodpercig, ha nincs semmi. Visszaad legfeljebb batch_size elemet.
        Ha timeout None, akkor blokkol, amíg legalább egy elem van.
        """
        with self.not_empty:
            if timeout is None:
                while len(self.items) == 0:
                    self.not_empty.wait()
            else:
                end = time.time() + timeout
                while len(self.items) == 0:
                    remaining = end - time.time()
                    if remaining <= 0:
                        return []
                    self.not_empty.wait(remaining)

            # most van legalább egy elem
            batch = []
            while len(batch) < batch_size and len(self.items) > 0:
                batch.append(self.items.popleft())
            return batch

    def clear(self):
        with self.lock:
            self.items.clear()


class CaptureThread(threading.Thread):
    def __init__(
            self,
            stop_event: threading.Event,
            source: int | str,
            buffer: SharedBuffer,
            camera_name: str = "DeafultCamera"
    ):

        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(source)
        self.buffer = buffer
        self.stop_event = stop_event
        self.frame_id = 0

        #For identify camera during stereo operation
        self.camera_name = camera_name

    def run(self):
        if not self.cap.isOpened():
            print(f"[Capture] -> Cannot open source {self.camera_name}")
            self.stop_event.set()
            return

        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                print("[Capture] -> No frame (end of video or camera error).")
                break

            self.frame_id += 1
            item = FrameItem(frame_id=self.frame_id, frame=frame, timestamp=time.time())
            self.buffer.push(item)

            # magas fps esetén hogy ne terheljük túl a buffert
            # time.sleep(0.001)

        self.cap.release()
        print("[Capture] stopped")
import threading, queue, traceback, cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


class DetectionThread(threading.Thread):
    def __init__(
        self,
        stop_event: threading.Event,
        shared_buffer,
        display_queue: queue.Queue,
        model_path: str,
        batch_size: int = 5,
        batch_timeout: float = 0.2,
        imgsz: int = 640,
        device: int | str = "cpu",
    ):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.shared_buffer = shared_buffer
        self.display_queue = display_queue
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.imgsz = imgsz
        self.device = device
        self.model_path = model_path

        # mindig YOLO modell
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print("[Detect] model load failed:", e)
            traceback.print_exc()
            self.model = None

        # annotátorok
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()

    def _rescale_boxes_to_orig(self, xyxy: np.ndarray, orig_w: int, orig_h: int) -> np.ndarray:
        if xyxy.size == 0:
            return xyxy
        scale_x = orig_w / float(self.imgsz)
        scale_y = orig_h / float(self.imgsz)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]] * scale_x
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]] * scale_y
        xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, orig_w - 1)
        xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, orig_h - 1)
        return xyxy

    def _parse_result_to_detections(self, res, orig_w, orig_h):
        detections = sv.Detections.from_ultralytics(res)
        detections.xyxy = self._rescale_boxes_to_orig(detections.xyxy, orig_w, orig_h)
        return detections

    def run(self):
        print("[Detect] started (YOLO only)")
        if self.model is None:
            print("[Detect] no model loaded, exiting")
            return

        while not self.stop_event.is_set():
            items = self.shared_buffer.pop_batch(self.batch_size, timeout=self.batch_timeout)
            if not items:
                continue

            batch_images = []
            for it in items:
                img_resized = cv2.resize(it.frame, (self.imgsz, self.imgsz))
                batch_images.append(img_resized)

            try:
                results = self.model.predict(batch_images, imgsz=self.imgsz, device=self.device, verbose=True)
            except Exception as e:
                print("[Detect] inference failed:", e)
                traceback.print_exc()
                for it in items:
                    if not self.display_queue.full():
                        self.display_queue.put((it.frame_id, it.frame))
                continue

            for it, res in zip(items, results):
                orig_h, orig_w = it.frame.shape[:2]
                dets = self._parse_result_to_detections(res, orig_w, orig_h)

                annotated = it.frame.copy()
                try:
                    annotated = self.box_annotator.annotate(annotated, dets)
                    labels = [f"{c}:{round(s,2)}" for c, s in zip(dets.class_id, dets.confidence)]
                    annotated = self.label_annotator.annotate(annotated, dets, labels)
                except Exception:
                    for (x1, y1, x2, y2), c in zip(dets.xyxy, dets.confidence):
                        cv2.rectangle(annotated, (int(round(x1)), int(round(y1))),
                                      (int(round(x2)), int(round(y2))), (0, 255, 0), 2)

                if not self.display_queue.full():
                    self.display_queue.put((it.frame_id, annotated))

        print("[Detect] stopped")

class DisplayThread(threading.Thread):
    def __init__(self, stop_event, display_queue, window_name="Pipeline"):
        super().__init__(daemon=True)
        self.stop_event = stop_event
        self.display_queue = display_queue
        self.window_name = window_name
        self.timestamps = deque(maxlen=30)   # rolling window a display FPS-hez
        self.fps_ema = 0.0
        self.ema_alpha = 0.2
        self.last_console_time = time.time()
        self.console_count = 0

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        while not self.stop_event.is_set():
            try:
                frame_id, frame = self.display_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            now = time.time()
            self.timestamps.append(now)
            self.console_count += 1

            # Instant FPS (lehet zajos)
            if len(self.timestamps) >= 2:
                fps_instant = 1.0 / (self.timestamps[-1] - self.timestamps[-2])
            else:
                fps_instant = 0.0

            # Rolling FPS
            if len(self.timestamps) >= 2:
                fps_rolling = len(self.timestamps) / (self.timestamps[-1] - self.timestamps[0])
            else:
                fps_rolling = fps_instant

            # EMA FPS
            self.fps_ema = self.ema_alpha * fps_instant + (1 - self.ema_alpha) * self.fps_ema

            # Overlay text
            text = f"FPS inst:{fps_instant:.1f} roll:{fps_rolling:.1f} ema:{self.fps_ema:.1f}"
            cv2.putText(frame, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            cv2.imshow(self.window_name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop_event.set()
                break

        cv2.destroyAllWindows()


class Pipeline:
    def __init__(self, source: int | str, model_path: str, device: int | str = "cpu"):
        self.shared_buffer = SharedBuffer(maxlen=512)
        self.display_queue: queue.Queue = queue.Queue(maxsize=64)
        self.stop_event = threading.Event()
        self.capture = CaptureThread(self.stop_event, source, self.shared_buffer)
        self.detect = DetectionThread(self.stop_event, self.shared_buffer, self.display_queue, model_path, batch_size=5, batch_timeout=0.15, imgsz=640, device=device)
        self.display = DisplayThread(self.stop_event, self.display_queue)

    def start(self):
        self.capture.start()
        self.detect.start()
        self.display.start()

    def join(self):
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()
        # make sure threads are stopped
        self.capture.join()
        self.detect.join()
        self.display.join()


if __name__ == "__main__":
    # példa indítás
    SRC = 0  # vagy "../videos/test.mp4"
    #SRC = "../sources/vid/speed_example_720p.mp4"  # vagy "../videos/test.mp4"
    MODEL_PATH = "../models/yolo11l.engine"  # vagy a te modell elérési útja
    #MODEL_PATH = "experiment-sxxxi/1"  # vagy a te modell elérési útja
    DEVICE = 0  # GPU: 0 vagy 'cuda:0', CPU: 'cpu'

    pipeline = Pipeline(source=SRC, model_path=MODEL_PATH, device=DEVICE)
    pipeline.start()
    pipeline.join()
