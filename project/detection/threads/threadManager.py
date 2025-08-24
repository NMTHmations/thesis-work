class ThreadManager:
    def __init__(self, queues : tuple, threads : tuple):
        self.queues = queues
        self.threads = threads

    def start(self):

        for t in self.threads:
            t.daemon = True
            t.start()

    def join(self):
        for t in self.threads:
            t.join()



