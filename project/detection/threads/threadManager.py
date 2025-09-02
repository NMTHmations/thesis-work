import time


class ThreadManager:
    def __init__(self, stopEvent,threads):
        self.stop_event = stopEvent
        self.threads = threads

    def start(self):
        for t in self.threads:
            t.start()

    def join(self):
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()
        # make sure threads are stopped
        for t in self.threads:
            t.join()

