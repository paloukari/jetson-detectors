# from http://blog.blitzblit.com/2017/12/24/asynchronous-video-capture-in-python-with-opencv/

import threading
import cv2


class VideoCaptureAsync:
    def __init__(self, src=0, width=640, height=480, resize_width=300, resize_height=300):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.frame_resized = []
        self.started = False
        self.read_lock = threading.Lock()
        self.resize_width = resize_width
        self.resize_height = resize_height

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchronous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.frame_resized = cv2.resize(
                    frame, (self.resize_width, self.resize_height))

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            frame_resized = self.frame_resized.copy()
            grabbed = self.grabbed

        return grabbed, frame, frame_resized

    def test(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()
