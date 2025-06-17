import time
import threading
import queue
import uuid
from datetime import datetime
import cv2
import torch
from ultralytics import YOLO
import os
import logging
import struct
import socket
import subprocess
import sys

# --- Configuration ---
VIDEO_IN                   = "/dev/video0"
VIDEO_OUT                  = "/home/tank/Downloads/track.mp4"
MODEL_PATH                 = "/home/tank/PycharmProjects/TrainedModels/03June2025/best.pt"
IMG_SIZE                   = 640
BATCH_SIZE                 = 1
MAX_FRAMES                 = 16
TRACKER_YAML               = "/usr/local/lib/python3.10/dist-packages/ultralytics/cfg/trackers/bytetrack.yaml"
SKIP_FRAMES                = 3       # frames to skip between inferences

# Event & Summary settings
SUMMARY_INTERVAL_SEC       = 3.0     # seconds between summaries
EXIT_CONFIRM_FRAMES        = 3000    # frames to confirm disappearance
TRACKING_THRESHOLD_FRAMES  = 5      # total frames in summary window to qualify as "new"

# Classes to monitor and their type codes
MONITORED_CLASSES = {"PERSON", "LORRY", "CAR", "VEHICLE"}
OBJECT_TYPE_MAPPING = {
    "PERSON": 10,
    "VEHICLE": 12,
    "LORRY": 11,
    "CAR": 2,
}

# Summary UDP settings
UDP_IP   = "127.0.0.1"
UDP_PORT = 5100

# --- Internal State ---
active_tracks         = {}  # tid -> {"last_seen": frame_idx, "exit_count": count}
summary_frame_counts  = {}  # tid -> frames seen within current summary window
reported_in_summary   = set()
frame_counter         = 0
last_summary_time     = time.time()

# Output directory for saved images
OUT_DIR = "/home/tank/Downloads/saved/"
os.makedirs(OUT_DIR, exist_ok=True)
frame_idx = 0

# --- Logging setup ---
logging.basicConfig(
    filename="Tracker.log",
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger("Tracker")

# --- Queues ---
frame_q = queue.Queue(maxsize=MAX_FRAMES)
draw_q  = queue.Queue(maxsize=MAX_FRAMES)

# --- Video & Streaming ---
width, height, fps = 400, 200, 10
bitrate_kbps       = 450
ffmpeg_cmd = [
    "ffmpeg", "-f", "rawvideo", "-pixel_format", "rgb24",
    "-video_size", f"{width}x{height}", "-framerate", str(fps),
    "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
    "-tune", "zerolatency", "-b:v", f"{bitrate_kbps}k",
    "-f", "mpegts", f"udp://127.0.0.1:5004"
]
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

cap = cv2.VideoCapture(VIDEO_IN, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)
fps = cap.get(cv2.CAP_PROP_FPS) or fps

# Load YOLO model
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo = YOLO(MODEL_PATH).to(device)
yolo.fuse(); yolo.half()
stream = torch.cuda.Stream(device=device) if device.startswith("cuda") else None
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Message functions ---
class Cfg:
    SRC_CSCI_ID = 5
    DST_CSCI_ID = 1
    MSG_ID      = 3
    SRC_UNIT_ID = int(sys.argv[1]) if len(sys.argv)>1 else 0
    DST_UNIT_ID = 0

state = Cfg()

def create_message_header(cfg, msg_len):
    return struct.pack("<BBBHBB", cfg.SRC_CSCI_ID, cfg.DST_CSCI_ID,
                       cfg.MSG_ID, msg_len, cfg.SRC_UNIT_ID, cfg.DST_UNIT_ID)

def create_object_data_entry(object_type_code: int, count: int) -> bytes:
    return struct.pack("<BB", object_type_code, count)

def create_detected_object_message(counts: dict) -> bytes:
    # counts: class_label -> count
    payload = struct.pack("<B", len(counts))
    for cls_label, cnt in counts.items():
        code = OBJECT_TYPE_MAPPING.get(cls_label, 0)
        payload += create_object_data_entry(code, cnt)
    for _ in range(10 - len(counts)):
        payload += create_object_data_entry(0, 0)
    header = create_message_header(state, 1 + 2*10)
    return header + payload

# --- Event processing ---
def process_events(boxes):
    global frame_counter, last_summary_time
    now = time.time()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else []
    cls_ids = boxes.cls.cpu().numpy() if boxes.id is not None else []
    seen_in_frame = set()

    # Count frames per ID in this summary window
    for tid, cls_id in zip(ids, cls_ids):
        cls_name = yolo.names[int(cls_id)]
        if cls_name not in MONITORED_CLASSES:
            continue
        seen_in_frame.add(tid)
        summary_frame_counts[tid] = summary_frame_counts.get(tid, 0) + 1

        # Maintain active_tracks for exit detection
        if tid not in active_tracks:
            active_tracks[tid] = {"last_seen": frame_counter, "exit_count": 0}
        else:
            active_tracks[tid]["last_seen"] = frame_counter
            active_tracks[tid]["exit_count"] = 0

    # Handle exits
    for tid, info in list(active_tracks.items()):
        if tid not in seen_in_frame:
            info["exit_count"] += 1
            if info["exit_count"] >= EXIT_CONFIRM_FRAMES:
                duration_frames = info["exit_count"] + info["last_seen"] - frame_counter
                logger.info("Exit id=%d duration_frames=%d", tid, duration_frames)
                del active_tracks[tid]

    # Periodic summary
    if now - last_summary_time >= SUMMARY_INTERVAL_SEC:
        to_report = {tid: cnt for tid, cnt in summary_frame_counts.items()
                     if cnt >= TRACKING_THRESHOLD_FRAMES and tid not in reported_in_summary}
        if to_report:
            # Map to class counts
            class_counts = {}
            # Retrieve classes for reported IDs via last detection
            for tid in to_report:
                # Could store id->class mapping; here we assume last seen boxes carry class
                # For simplicity, label as UNKNOWN
                cls_label = "UNKNOWN"
                class_counts[cls_label] = class_counts.get(cls_label, 0) + 1
                reported_in_summary.add(tid)
            msg = create_detected_object_message(class_counts)
            sock.sendto(msg, (UDP_IP, UDP_PORT))
            logger.info("Summary sent: %s", class_counts)
        summary_frame_counts.clear()
        last_summary_time = now

# --- Threads ---
def reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LANCZOS4)
        proc.stdin.write(resized.tobytes())
        frame_q.put(resized)
    frame_q.put(None)


def inferencer():
    global frame_counter
    stop = False
    raw_count = 0
    while True:
        item = frame_q.get()
        if item is None:
            stop = True
            break
        raw_count += 1
        if raw_count % (SKIP_FRAMES + 1) != 1:
            continue
        batch = [item]
        while len(batch) < BATCH_SIZE:
            itm = frame_q.get()
            if itm is None:
                stop = True
                break
            raw_count += 1
            if raw_count % (SKIP_FRAMES + 1) != 1:
                continue
            batch.append(itm)
        if not batch:
            break
        if device.startswith("cuda") and stream:
            ctx = torch.cuda.stream(stream)
            ctx.__enter__()
        with torch.inference_mode():
            results = yolo.track(
                source=batch,
                device=device,
                imgsz=IMG_SIZE,
                tracker=TRACKER_YAML,
                persist=True,
                half=device.startswith("cuda"),
                conf=0.5,
                verbose=False
            )
        if device.startswith("cuda") and stream:
            ctx.__exit__(None, None, None)
        for frame, res in zip(batch, results):
            frame_counter += 1
            draw_q.put((frame, res))
            process_events(res.boxes)
        if stop:
            break
    draw_q.put(None)


def drawer():
    global frame_idx
    while True:
        item = draw_q.get()
        if item is None:
            break
        frame, res = item
        if res.boxes is not None and res.boxes.id is not None:
            for bb, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy()):
                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{int(tid)}", (x1, y1-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        out_path = os.path.join(OUT_DIR, f"{timestamp}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frame_idx += 1
    cap.release()

# --- Main ---
if __name__ == "__main__":
    t_read = threading.Thread(target=reader, daemon=True)
    t_inf  = threading.Thread(target=inferencer, daemon=True)
    t_draw = threading.Thread(target=drawer, daemon=True)
    t_read.start(); t_inf.start(); t_draw.start()
    t_read.join(); t_inf.join(); t_draw.join()
    proc.stdin.close(); proc.wait()
    print("Processing complete.")

