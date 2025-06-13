import datetime
import logging
import os
import socket
import struct
import subprocess
import uuid
import xml.etree.ElementTree as ET
import threading
import queue
import time  # for timing
import cv2
import numpy as np
import setproctitle
from torch.cuda import is_available as check_gpu
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
# from datetime import datetime

# -------------------------------------------------------------
# (0) Configuration flags
# -------------------------------------------------------------
DEBUGGING = True                        # Set False to suppress all logs
TIME_ANALYSIS = True                   # Set False to disable timing logs

# -------------------------------------------------------------
# (1) Basic setup: device, process name, logging, config reading
# -------------------------------------------------------------
device = 'cuda' if check_gpu() else 'cpu'
print("device is:", device)

setproctitle.setproctitle("yolo-object-tracker")

LOG_FILE = "tracker_errors.log"
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.DEBUG if DEBUGGING else logging.CRITICAL,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_FILE, mode='a')
    ]
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG if DEBUGGING else logging.CRITICAL)

# Read XML configuration
tree = ET.parse("config2.xml")
root = tree.getroot()
get_config = lambda tag, default=None: root.find(tag).text if root.find(tag) is not None else default

YOLO_MODEL = get_config("YOLO_MODEL")
VIDEO_SOURCE = get_config("VIDEO_SOURCE")
UDP_IP = get_config("UDP_IP")
UDP_PORT = int(get_config("UDP_PORT", "0"))
SAVE_PATH = get_config("SAVE_PATH")
THRESH_TIME_SEC_OBJECT_LOCATION = int(get_config("THRESH_TIME_SEC_OBJECT_LOCATION", "0"))
SKIP_FRAMES = int(get_config("SKIP_FRAMES", "1"))
MSG_ID = int(get_config("MSG_ID", "0"))
SRC_CSCI_ID = int(get_config("SRC_CSCI_ID", "0"))
DST_CSCI_ID = int(get_config("DST_CSCI_ID", "0"))
SRC_UNIT_ID = int(get_config("SRC_UNIT_ID", "0"))
DST_UNIT_ID = int(get_config("DST_UNIT_ID", "0"))

# Override flags if set in XML
if root.find("DEBUGGING") is not None:
    DEBUGGING = root.find("DEBUGGING").text == '1'
    logger.setLevel(logging.DEBUG if DEBUGGING else logging.CRITICAL)
if root.find("TIME_ANALYSIS") is not None:
    TIME_ANALYSIS = root.find("TIME_ANALYSIS").text == '1'

os.makedirs(SAVE_PATH, exist_ok=True)

# Shared UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Load YOLO model once
try:
    model = YOLO(YOLO_MODEL)
except Exception as e:
    logger.error(f"Model load failed: {e}")
    raise


# Heatup / Warm-up model
try:
    dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)  # Black image
    model.predict(dummy_input, verbose=False, conf=0.25,task='detect')   # One dummy prediction
    logger.info("Model heatup complete.")
except Exception as e:
    logger.warning(f"Model heatup failed: {e}")



# ThreadPoolExecutor for nonblocking UDP
executor = ThreadPoolExecutor(max_workers=2)

# -------------------------------------------------------------
# (2) Globals for display
# -------------------------------------------------------------
WIDTH = 0
HEIGHT = 0
DISPLAY_DELAY = 0.0    # seconds between frames

# -------------------------------------------------------------
# (3) Object-type mapping for monitored classes
# -------------------------------------------------------------
OBJECT_TYPE_MAPPING = {
    "PERSON": 10,
    "VEHICLE": 12,
    # "MOTORCYCLE": 12,
}
MONITORED_CLASSES = set(OBJECT_TYPE_MAPPING.keys())

# -------------------------------------------------------------
# (4) Message struct packing helpers
# -------------------------------------------------------------
def create_message_header_new(msg_len: int) -> bytes:
    return struct.pack(
        '<BBBHBB',
        SRC_CSCI_ID,
        DST_CSCI_ID,
        MSG_ID,
        msg_len,
        SRC_UNIT_ID,
        DST_UNIT_ID
    )

def create_object_data_entry(object_type_code: int, count: int) -> bytes:
    return struct.pack('<BB', object_type_code, count)

def create_detected_object_message(objects_info: list) -> bytes:
    entries = []
    for obj in objects_info:
        lbl = obj['class_label']
        cnt = obj['count']
        if lbl not in OBJECT_TYPE_MAPPING:
            continue
        entries.append((OBJECT_TYPE_MAPPING[lbl], cnt))
    entries = entries[:10]
    payload = struct.pack('<B', len(entries))
    for code, cnt in entries:
        payload += create_object_data_entry(code, cnt)
    for _ in range(10 - len(entries)):
        payload += create_object_data_entry(0, 0)
    header = create_message_header_new(len(payload))
    return header + payload

# -------------------------------------------------------------
# (5) Queues and threading state
# -------------------------------------------------------------
ip_queue = queue.Queue(maxsize=20)
processed_queue = queue.Queue(maxsize=15)
infer_queue = queue.Queue(maxsize=2)
save_queue = queue.Queue(maxsize=30)
send_queue = queue.Queue(maxsize=20)
display_queue = queue.Queue(maxsize=10)
stop_event = threading.Event()

# -------------------------------------------------------------
# (6) Timing utilities
# -------------------------------------------------------------
def log_time(thread_name, elapsed, frame_id=None):
    if TIME_ANALYSIS:
        msg = f"TIMING [{thread_name}]"
        if frame_id is not None:
            msg += f" frame={frame_id}"
        msg += f" elapsed={elapsed*1000:.2f}ms"
        logger.info(msg)

# -------------------------------------------------------------
# (7) Preprocessing functions
# -------------------------------------------------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def preprocess_frame(frame):
    global clahe
    try:
        # # Upload frame to GPU
        # gpu_frame = cv2.cuda_GpuMat()
        # gpu_frame.upload(frame)
        #
        # # Apply bilateral filter on GPU
        # gpu_filtered = cv2.cuda.bilateralFilter(gpu_frame, d=9, sigmaColor=75, sigmaSpace=75)
        #
        # # Download the result back to CPU
        # bilateral = gpu_filtered.download()
        return frame
        bilateral = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        lab = cv2.cvtColor(bilateral, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_cl = clahe.apply(l)
        lab_cl = cv2.merge((l_cl, a, b))
        contrast = cv2.cvtColor(lab_cl, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(contrast, (9,9), sigmaX=10)
        sharpened = cv2.addWeighted(contrast, 1.5, blurred, -0.5, 0)
        return sharpened
    except Exception as e:
        logger.error(f"Preproc error: {e}")
        return frame


def preprocess_frame_parallel(frame, max_workers=4):
    h, w = frame.shape[:2]
    mid_h, mid_w = h // 2, w // 2
    quads = [
        frame[0:mid_h,    0:mid_w],
        frame[0:mid_h,    mid_w:w],
        frame[mid_h:h,    0:mid_w],
        frame[mid_h:h,    mid_w:w],
    ]
    with ThreadPoolExecutor(max_workers=max_workers) as execp:
        futures = [execp.submit(preprocess_frame, q) for q in quads]
        processed_quads = [f.result() for f in futures]
    top = np.hstack((processed_quads[0], processed_quads[1]))
    bottom = np.hstack((processed_quads[2], processed_quads[3]))
    return np.vstack((top, bottom))

# -------------------------------------------------------------
# (8) Thread functions
# -------------------------------------------------------------
def capture_thread_func():
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    frame_count = 0
    frame_id = 0
    if not cap.isOpened():
        logger.error("Capture: Could not open source")
        stop_event.set()
        return
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        start = time.perf_counter()
        height, width = frame.shape[:2]
        left = int(width * 0.3)
        right = int(width * 0.7)

        # Crop 30% from top and bottom
        top = int(height * 0)
        bottom = int(height * 1)

        frame = frame[top:bottom, left:right]
        if frame_count % SKIP_FRAMES != 0:
            frame_count = (frame_count + 1) % SKIP_FRAMES
            continue
        frame_count = (frame_count + 1) % SKIP_FRAMES
        try:
            ip_queue.put((frame_id, frame), timeout=1)
        except queue.Full:
            logger.warning("Capture: ip_queue full")
        elapsed = time.perf_counter() - start
        log_time('capture', elapsed, frame_id)
        frame_id += 1
    cap.release()


def image_processing_thread_func():
    while not stop_event.is_set():
        try:
            frame_id, frame = ip_queue.get(timeout=1)
        except queue.Empty:
            continue
        start = time.perf_counter()
        processed = preprocess_frame_parallel(frame)
        try:
            processed_queue.put((frame_id, processed), timeout=1)
        except queue.Full:
            logger.warning("ImageProc: processed_queue full")
        elapsed = time.perf_counter() - start
        log_time('imageproc', elapsed, frame_id)


def process_thread_func():
    prediction_buffer = []
    prev_counts = {}
    frequent_alert = {}
    while not stop_event.is_set():
        try:
            frame_id, proc = processed_queue.get(timeout=1)
        except queue.Empty:
            continue
        start = time.perf_counter()
        infer_queue.put((frame_id, proc))
        fid, frame = infer_queue.get()
        try:
            results = model.predict(
                source=frame,
                verbose=False,
                task='detect',
                # device=0,
                # imgsz=640,
                conf=0.70,
                show=False,
                # half=True
            )
        except Exception as e:
            logger.error(f"Process: YOLO inference failed: {e}")
            continue
        detections = []
        if results and hasattr(results[0], 'boxes') and results[0].boxes is not None:
            boxes = results[0].boxes.data.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2, conf, cls = box
                label = model.names[int(cls)]
                detections.append({'class_label': label, 'bbox': (x1,y1,x2,y2)})
        prediction_buffer.append(detections)
        agg = {}
        for preds in prediction_buffer:
            for det in preds:
                agg.setdefault(det['class_label'], []).append(det)
        current_counts = {lbl: len(lst) for lbl, lst in agg.items()}
        final_detections = {}
        now_ts = datetime.datetime.now()
        for lbl in MONITORED_CLASSES:
            freq = current_counts.get(lbl, 0)
            last = prev_counts.get(lbl, 0)
            last_alert = frequent_alert.get(lbl)
            if freq > last and last>0 and ((last_alert is None)  or ((now_ts - last_alert).total_seconds() > THRESH_TIME_SEC_OBJECT_LOCATION)):
                final_detections[lbl] = {'frequency': freq, 'bbox': [d['bbox'] for d in agg[lbl]]}
                frequent_alert[lbl] = now_ts
            prev_counts[lbl] = freq
        if not final_detections:
            prediction_buffer.clear()
            elapsed = time.perf_counter() - start
            log_time('process', elapsed, frame_id)
            continue
        annotated = frame.copy()
        for res in results:
            annotated = res.plot()
        objects_info = [{'class_label': k, 'count': v['frequency']} for k,v in final_detections.items()]
        # filename = f"{uuid.uuid4().hex[:21]}.jpg"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        filename = f"{timestamp}.jpg"
        try:
            save_queue.put((frame_id, {'annotated_frame': annotated, 'objects_info': objects_info, 'filename': filename}), timeout=1)
        except queue.Full:
            logger.warning("Process: save_queue full")
        elapsed = time.perf_counter() - start
        log_time('process', elapsed, frame_id)
        prediction_buffer.clear()


def save_thread_func():
    while not stop_event.is_set() or not save_queue.empty():
        try:
            frame_id, item = save_queue.get(timeout=1)
        except queue.Empty:
            continue
        start = time.perf_counter()
        frame_path = os.path.join(SAVE_PATH, item['filename'])
        try:
            cv2.imwrite(frame_path, item['annotated_frame'])
        except Exception as e:
            logger.error(f"Save: Failed to write '{item['filename']}': {e}")
            continue
        try:
            send_queue.put((frame_id, {'filename': item['filename'], 'objects_info': item['objects_info']}), timeout=1)
        except queue.Full:
            logger.warning(f"Save: send_queue full; dropping {item['filename']}")
        elapsed = time.perf_counter() - start
        log_time('save', elapsed, frame_id)


def send_thread_func():
    while not stop_event.is_set() or not send_queue.empty():
        try:
            frame_id, item = send_queue.get(timeout=1)
        except queue.Empty:
            continue
        start = time.perf_counter()
        try:
            msg = create_detected_object_message(item['objects_info'])
            sock.sendto(msg, (UDP_IP, UDP_PORT))
            logger.info(f"Send: UDP packet for '{item['filename']}' sent ({len(msg)} bytes)")
        except Exception as e:
            logger.error(f"Send: Error sending UDP for '{item['filename']}': {e}")
        elapsed = time.perf_counter() - start
        log_time('send', elapsed, frame_id)

# -------------------------------------------------------------
# (9) Main entry
# -------------------------------------------------------------
def main():
    threads = [
        threading.Thread(target=capture_thread_func, name="CaptureThread", daemon=True),
        threading.Thread(target=image_processing_thread_func, name="ImageProcThread", daemon=True),
        threading.Thread(target=process_thread_func, name="ProcessThread", daemon=True),
        threading.Thread(target=save_thread_func, name="SaveThread", daemon=True),
        threading.Thread(target=send_thread_func, name="SendThread", daemon=True)
    ]
    for t in threads:
        t.start()
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Main: KeyboardInterrupt received — shutting down.")
        stop_event.set()
    for t in threads:
        t.join(timeout=2)
    sock.close()
    executor.shutdown(wait=False)
    logger.info("Program terminated.")

if __name__ == "__main__":
    main()
