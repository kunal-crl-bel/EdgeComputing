import threading
import cv2
import logging
import os
import queue

# before any VideoCapture calls, lock FFmpeg to 1 thread
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

class VideoStream:
    def __init__(self, src, queue_size=5):
        self.src = src
        self.q = queue.Queue(maxsize=queue_size)
        self.stopped = threading.Event()
        self.thread = None

    def start(self):
        if self.thread is None:
            self.thread = threading.Thread(target=self._worker, daemon=True)
            self.thread.start()
            logging.info("VideoStream worker thread started.")
        return self

    def _worker(self):
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {self.src}")
            logging.info(f"Opened video source {self.src}")
        except Exception:
            logging.exception("Failed to open VideoCapture")
            self.stopped.set()
            return

        while not self.stopped.is_set():
            try:
                ret, frame = cap.read()
                if not ret:
                    logging.warning("Stream ended or read error; stopping")
                    break 

                # block if queue is full, but wake up to check `stopped`
                while not self.stopped.is_set():
                    try:
                        self.q.put(frame, timeout=0.1)
                        break
                    except queue.Full:
                        continue

            except cv2.error:
                logging.exception("OpenCV/FFmpeg error in worker; stopping")
                break
            except Exception:
                logging.exception("Unexpected error in worker; stopping")
                break

        cap.release()
        logging.info("VideoCapture released")
        self.stopped.set()

    def read(self, timeout=None):
        """
        Returns next frame or None if stream has ended.
        If timeout is set, blocks up to timeout seconds.
        """
        try:
            return self.q.get(timeout=timeout)
        except queue.Empty:
            return None

    def more(self):
        """
        Returns True as long as there might still be frames
        (either in the queue or the worker is still running).
        """
        return not self.q.empty() or not self.stopped.is_set()

    def stop(self):
        self.stopped.set()
        if self.thread:
            self.thread.join()
        logging.info("VideoStream stopped")
