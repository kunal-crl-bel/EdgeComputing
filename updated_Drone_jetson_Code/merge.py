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
from PIL import Image
import sys

# --- Configuration ---
VIDEO_IN     = "/dev/video0"
VIDEO_OUT    = "/home/tank/Downloads/track.mp4"
MODEL_PATH   = "/home/tank/PycharmProjects/TrainedModels/03June2025/best.pt"
IMG_SIZE     = 640
BATCH_SIZE   = 1
MAX_FRAMES   = 16
TRACKER_YAML = "/usr/local/lib/python3.10/dist-packages/ultralytics/cfg/trackers/bytetrack.yaml"
SKIP_FRAMES  = 3  # Number of frames to skip between inferences

# Event & Summary settings
SUMMARY_INTERVAL    = 5.0   # seconds between summaries
EXIT_CONFIRM_FRAMES = 3000  # frames to confirm disappearance

# Classes to monitor and their type codes
MONITORED_CLASSES = {"PERSON", "LORRY", "CAR", "VEHICLE"}
OBJECT_TYPE_MAPPING = {
    "PERSON": 10,
    "VEHICLE": 12,
    "LORRY": 11,
    "CAR": 2,
}

# Summary UDP settings
UDP_IP = "192.168.171.10"
UDP_PORT = 5100

class Cfg:
    SRC_CSCI_ID = 5
    DST_CSCI_ID = 1
    MSG_ID      = 3
    SRC_UNIT_ID = int(sys.argv[1])
    DST_UNIT_ID = 2

class State:
    def __init__(self):
        self.cfg = Cfg
        self.OBJECT_TYPE_MAPPING = OBJECT_TYPE_MAPPING

state = State()

# --- State for events ---
active_tracks     = {}   # tid -> track info
recent_entries    = set()
seen_ids_ever     = set()
first_seen_ts     = {}
last_summary_time = time.time()
frame_counter     = 0

# Output directory for saving frames (optional)
OUT_DIR = "/home/tank/Downloads/saved/"
os.makedirs(OUT_DIR, exist_ok=True)

# Frame index for saved images
frame_idx = 0

# --- Logging setup ---
LOG_FILE = "Tracker.log"
logging.basicConfig(
    filename=LOG_FILE,
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)
logger = logging.getLogger("Tracker")

# --- Queues ---
frame_q = queue.Queue(maxsize=MAX_FRAMES)
draw_q  = queue.Queue(maxsize=MAX_FRAMES)

# --- Video & Streaming setup ---
width, height, fps = 400,200, 10
bitrate_kbps      = 450
# Live stream destination
dst_ip, dst_port  = "127.0.0.1", 5004

# Build FFmpeg command to stream raw BGR frames
ffmpeg_cmd = [
    "ffmpeg",
    "-f", "rawvideo",
    "-pixel_format", "rgb24",
    "-video_size", f"{width}x{height}",
    "-framerate", str(fps),
    "-i", "-",                 # stdin
    "-c:v", "libx264",
    "-preset", "ultrafast",
    "-tune", "zerolatency",
    "-b:v", f"{bitrate_kbps}k",
    "-f", "mpegts",
    f"udp://{dst_ip}:{dst_port}"
]
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

# Capture and file writer
cap = cv2.VideoCapture(VIDEO_IN, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, fps)

fps   = cap.get(cv2.CAP_PROP_FPS) or fps
W, H  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#writer = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (W, H))

# --- Load YOLO model ---
cuda_available = torch.cuda.is_available()
device = "cuda" if cuda_available else "cpu"
yolo = YOLO(MODEL_PATH).to(device)
yolo.fuse()
yolo.half()
if cuda_available:
    #yolo.to(device)
    #stream=None
    stream = torch.cuda.Stream(device=device)
    print(f"Using GPU on {device}")
else:
    stream = None
    print("CUDA not available, using CPU")
    
 # --- UDP socket for summary messages ---
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- Message creation functions ---
def create_message_header_new(cfg, msg_len: int) -> bytes:
    return struct.pack(
        "<BBBHBB",
        cfg.SRC_CSCI_ID,
        cfg.DST_CSCI_ID,
        cfg.MSG_ID,
        msg_len,
        cfg.SRC_UNIT_ID,
        cfg.DST_UNIT_ID
    )

def create_object_data_entry(object_type_code: int, count: int) -> bytes:
    return struct.pack("<BB", object_type_code, count)

def create_detected_object_message(state, objects_info: list) -> bytes:
    entries = []
    for obj in objects_info:
        lbl = obj["class_label"]
        cnt = obj["count"]
        if lbl not in state.OBJECT_TYPE_MAPPING:
            continue
        code = state.OBJECT_TYPE_MAPPING[lbl]
        entries.append((code, cnt))
    entries = entries[:10]
    no_of_objects = len(entries)
    payload = struct.pack("<B", no_of_objects)
    for code, cnt in entries:
        payload += create_object_data_entry(code, cnt)
    for _ in range(10 - no_of_objects):
        payload += create_object_data_entry(0, 0)
    msg_len = 1 + (2 * 10)
    header = create_message_header_new(state.cfg, msg_len)
    return header + payload
    
    
    

# --- Event processing ---
def process_events(boxes):
    global last_summary_time, frame_counter
    now = time.time()
    ids = boxes.id.cpu().numpy() if boxes.id is not None else []
    cls_ids = boxes.cls.cpu().numpy()
    seen_ids_in_frame = set()

    # Entry / re-entry detection
    for tid, cls_id in zip(ids, cls_ids):
        cls_name = yolo.names[int(cls_id)]
        if cls_name not in MONITORED_CLASSES:
            continue
        seen_ids_in_frame.add(tid)

        if tid not in active_tracks:
            if tid not in seen_ids_ever:
                seen_ids_ever.add(tid)
                first_seen_ts[tid] = now
                active_tracks[tid] = {
                    "class": cls_name,
                    "entry_ts": first_seen_ts[tid],
                    "last_seen_frame": frame_counter,
                    "exit_count": 0
                }
                recent_entries.add(tid)
                ts = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
                logger.info(f"Entry detected: id=%d class=%s timestamp=%s", int(tid), cls_name, ts)
            else:
                entry_ts = first_seen_ts.get(tid, now)
                active_tracks[tid] = {
                    "class": cls_name,
                    "entry_ts": entry_ts,
                    "last_seen_frame": frame_counter,
                    "exit_count": 0
                }
                ts = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
                logger.info(f"Reappearance detected: id=%d class=%s timestamp=%s", int(tid), cls_name, ts)
        else:
            active_tracks[tid]["last_seen_frame"] = frame_counter
            active_tracks[tid]["exit_count"] = 0

    # Exit detection
    for tid, info in list(active_tracks.items()):
        if tid not in seen_ids_in_frame:
            info["exit_count"] += 1
            if info["exit_count"] >= EXIT_CONFIRM_FRAMES:
                duration = round(now - info["entry_ts"], 3)
                ts = datetime.fromtimestamp(now).isoformat(timespec="milliseconds")
                logger.info(f"Exit detected: id=%d class=%s timestamp=%s duration=%.3f", int(tid), info["class"], ts, duration)
                del active_tracks[tid]

    # Periodic Summary only if new entries in interval
    if now - last_summary_time >= SUMMARY_INTERVAL:
        if recent_entries:
            # Count per class
            counts = {}
            for tid in recent_entries:
                cls = active_tracks.get(tid, {}).get("class")
                if cls:
                    counts[cls] = counts.get(cls, 0) + 1
            objects_info = []
            for cls_label, cnt in counts.items():
                objects_info.append({"class_label": cls_label, "count": cnt})
            # Build and send message
            msg_bytes = create_detected_object_message(state, objects_info)
            sock.sendto(msg_bytes, (UDP_IP, UDP_PORT))
            logger.info("Summary sent: %s", counts)
            recent_entries.clear()
        last_summary_time = now

# --- Threads ---
def reader():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        new_size = (width,height)
        resized_rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_LANCZOS4)
        proc.stdin.write(resized_rgb.tobytes())
        # enqueue for inference/drawing
        frame_q.put(resized_rgb)
        #continue
        
        
        # Bayer→RGB
        #gray = frame[:,:,0] if frame.ndim==3 else frame
        #rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # convert to BGR for streaming
        #bgr  = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        # non-blocking send raw frame to FFmpeg
        #proc.stdin.write(rgb.tobytes())
        # enqueue for inference/drawing
        #frame_q.put(rgb)
    frame_q.put(None)

def inferencer():
    global frame_counter
    stop = False
    raw_count = 0
    while True:
        rgb = frame_q.get()
        item  = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if item is None:
            stop = True
        else:
            raw_count += 1
            if raw_count % (SKIP_FRAMES + 1) != 1:
                continue
        batch = []
        if not stop:
            batch.append(item)
        while not stop and len(batch) < BATCH_SIZE:
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
        if cuda_available and stream:
            ctx = torch.cuda.stream(stream)
            ctx.__enter__()
        with torch.inference_mode():
            results = yolo.track(
                source=batch,
                device=device,
                imgsz=IMG_SIZE,
                tracker=TRACKER_YAML,
                persist=True,
                half=cuda_available,
                verbose=False,
                conf=.5,
                #show=True
            )
            
        if cuda_available and stream:
            ctx.__exit__(None, None, None)
        frame_counter += 1
        #print(results)
        for frame, res in zip(batch, results):
           
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
            xyxy = res.boxes.xyxy.cpu().numpy()
            ids  = res.boxes.id.cpu().numpy()
            for bb, tid in zip(xyxy, ids):
                x1, y1, x2, y2 = map(int, bb)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"ID:{int(tid)}",
                    (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        #timestamp="img0"
        filename = f"{timestamp}.jpg"
        out_path = os.path.join(OUT_DIR, filename)

        cv2.imwrite(out_path, frame)
        frame_idx += 1
    cap.release()

# --- Main ---
if __name__ == "__main__":
    # start threads
    t_read = threading.Thread(target=reader, daemon=True)
    t_inf  = threading.Thread(target=inferencer, daemon=True)
    t_draw = threading.Thread(target=drawer, daemon=True)
    t_read.start(); t_inf.start(); t_draw.start()
    t_read.join(); t_inf.join(); t_draw.join()
    # cleanup streaming
    proc.stdin.close(); proc.wait()
    #writer.release(); sock.close()
    print("Processing complete.")

