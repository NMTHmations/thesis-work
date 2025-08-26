import threading
from collections import deque
import time
from typing import Deque, Optional, List

from project.detection.types.FrameItem import FrameItem


class FrameBuffer:

    def __init__(self, maxLength: int = 512):
        #Deque for O(1) methods
        self.items: Deque[FrameItem] = deque(maxlen=maxLength)
        self.lock = threading.Lock()
        self.notEmpty = threading.Condition(self.lock)

    def push(self, item: FrameItem):
        with self.lock:

            #A legrégebbi frameket eldobjuk
            if len(self.items) == self.items.maxlen:
                self.items.popleft()

            self.items.append(item)
            self.notEmpty.notify()

    def pop_batch(self, batch_size: int, timeout: Optional[float] = None) -> List[FrameItem]:

        with self.notEmpty:
            if timeout is None:
                while len(self.items) == 0:
                    self.notEmpty.wait()
            else:
                end = time.time() + timeout
                while len(self.items) == 0:
                    remaining = end - time.time()
                    if remaining <= 0:
                        return []
                    self.notEmpty.wait(remaining)

            # most van legalább egy elem
            batch = []
            while len(batch) < batch_size and len(self.items) > 0:
                batch.append(self.items.popleft())
            return batch

    def clear(self):
        with self.lock:
            self.items.clear()

