import threading
from collections import deque
import time
from typing import Deque, Optional, List

from project.detection.types.FrameItem import FrameItem


class FrameBuffer:
    """
    A class representing a thread safe data structure to store video frames.
    """
    def __init__(self, maxLength: int = 128):
        #Deque for O(1) methods
        self.items: Deque[FrameItem] = deque(maxlen=maxLength)
        self.lock = threading.Lock()
        self.notEmpty = threading.Condition(self.lock)
        self.timeout : Optional[float] = None

    def push(self, item: FrameItem):
        with self.lock:

            #Dropping the oldest frames
            if len(self.items) == self.items.maxlen:
                self.items.popleft()

            self.items.append(item)
            self.notEmpty.notify()

    def pop_batch(self, batch_size: int) -> List[FrameItem]:

        with self.notEmpty:
            if self.timeout is None:
                while len(self.items) == 0:
                    self.notEmpty.wait()
            else:
                end = time.time() + self.timeout
                while len(self.items) == 0:
                    remaining = end - time.time()
                    if remaining <= 0:
                        return []
                    self.notEmpty.wait(remaining)

            #Now we have at least one element
            batch = []
            while len(batch) < batch_size and len(self.items) > 0:
                batch.append(self.items.popleft())
            return batch

    def clear(self):
        with self.lock:
            self.items.clear()

