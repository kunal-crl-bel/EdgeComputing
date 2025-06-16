import os
import cv2
import torch
import time
import queue
import threading
import numpy as np
from ultralytics import YOLO
from PIL import Image

from config import load_config
from db_manager import DBManager
from similarity_checker import SimilarityChecker
from notifier import Notifier
from VideoStream import VideoStream

# Global sets for tracking reported IDs and their timestamps
reported_ids = set()
last_seen = dict()
frame_map = dict()

# This mapping will be loaded from config based on model.names
CLASS_ID_TO_SRS = {}


def build_class_id_to_srs_map(model, cfg):
    """
    Builds a mapping from YOLO class index to SRS code using class names from the model
    and srs_values_of_classes from config.xml.
    """
    srs_config = cfg.get("srs_values_of_classes", {})
    srs_config2 = {}
    for i in srs_config:
        srs_config2[i.lower()] = srs_config[i]
        
    print('SRS config is : ',srs_config)
    mapping = {}
    for idx, name in model.names.items():
        print(f"Mapping class {idx} ({name}) to SRS code")
        if name.lower() in srs_config2:
            print("\n[ Found ] SRS mapping for class:", name,'\n')
            try:
                mapping[idx] = int(srs_config2[name.lower()])
            except ValueError:
                mapping[idx] = 0
        else:
            mapping[idx] = 0  # Default if not found
    return mapping


def get_srs_code(class_id):
    """
    Maps YOLO class IDs to corresponding SRS object type codes.
    Args:
        class_id (int): The class index from YOLO prediction.
    Returns:
        int: Mapped SRS code, or 0 if unmapped.
    """
    return CLASS_ID_TO_SRS.get(class_id, 0)


def detection_worker(cfg, model, stream, detection_queue, device):
    """
    Performs object detection and tracking, and enqueues valid object crops with metadata.
    Args:
        cfg (dict): Configuration dictionary.
        model: YOLO detection model.
        stream: Video stream object.
        detection_queue (Queue): Thread-safe queue for detections.
        device (str): 'cuda' or 'cpu'.
    """
    print("[Detection] Started")
    frame_id = 0
    global frame_map
    
    while stream.more():
        frame = stream.read()
        if frame is None:
            continue

        with torch.amp.autocast('cuda', enabled=True):
            results = model.track(source=frame,
                                  stream=True, # to tell ultralytics that i want to use same tracker.
                                  device=device,
                                  imgsz=640,
                                  conf=float(cfg['detect_confidence']),
                                  tracker="bytetrack.yaml",
                                  verbose=False)
            
            print(results)
            exit(0)

        annotated_frame = results.plot()
        frame_map[frame_id] = annotated_frame
        
        for box in results.boxes:
            tid = int(box.id)
            class_id = int(box.cls)

            if tid in reported_ids:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            h, w, _ = frame.shape
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = frame[y1:y2, x1:x2]
            detection_queue.put(
                (tid, crop, class_id, frame_id))             
            last_seen[tid] = time.time()
        
        frame_id += 1


def similarity_worker(cfg, detection_queue, notify_queue, sim_checker):
    """
    Compares new detections to known database entries to filter redundant objects.
    Args:
        cfg (dict): Configuration dictionary.
        detection_queue (Queue): Queue from detection stage.
        notify_queue (Queue): Queue to push new objects for notification.
        sim_checker (SimilarityChecker): Object similarity checker.
    """
    print("[Similarity] Started")
    while True:
        try:
            tid, crop, class_id, frame_id = detection_queue.get(timeout=1)
            is_new, hash_value = sim_checker.is_new(Image.fromarray(crop))
            if is_new:
                notify_queue.put((
                        tid, 
                        hash_value, 
                        crop, 
                        class_id, 
                        frame_id, 
                    ))
        except queue.Empty:
            time.sleep(0.1)  # Avoid busy waiting
            continue


def notifier_worker(cfg, notify_queue, db, notifier):
    """
    Sends SRS-coded notifications for unseen objects and persists them.
    Args:
        cfg (dict): Configuration dictionary.
        notify_queue (Queue): Queue containing unique, new detections.
        db (DBManager): Database for saving crops.
        notifier (Notifier): Sends batched UDP messages with SRS codes.
    """
    print("[Notifier] Started")
    global reported_ids
    saved_dir = cfg.get('saved_frame_dir', 'saved_frames')
    os.makedirs(saved_dir, exist_ok=True)


    while True:
        try:
            tid, hash_value, crop, class_id, frame_id = notify_queue.get(timeout=1)
            db.add_image(Image.fromarray(crop), id_hash=hash_value)
            srs_code = get_srs_code(class_id)
            notifier.add(srs_code, 1)
            reported_ids.add(tid)

            filename = os.path.join(saved_dir, f"frame_{frame_id}.jpg")
            cv2.imwrite(filename, frame_map[frame_id])
            print(f"[Notifier] New object {tid} with SRS {srs_code} saved to {filename}")
        except queue.Empty:
            time.sleep(0.1)  # Avoid busy waiting
            continue


def main():
    """
    System initializer and thread starter. Loads model, config, DB, and kicks off all workers.
    """
    import warnings
    warnings.filterwarnings(
        "ignore", category=UserWarning, module='torchvision')
    import sys
    if not sys.warnoptions:
        warnings.simplefilter("ignore")

    global reported_ids, last_seen, CLASS_ID_TO_SRS
    print("[Main] Initializing...")
    cfg = load_config("config.xml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(cfg.get('saved_frame_dir', 'saved_frames'), exist_ok=True)

    model = YOLO(cfg['model_path']).to(device)
    model.fuse()
    model.model.half()

    # Build SRS mapping at startup
    CLASS_ID_TO_SRS = build_class_id_to_srs_map(model, cfg)

    print("[Main] Warming up model...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    with torch.amp.autocast('cuda', enabled=True):
        _ = model.predict(source=dummy, device=device,
                          imgsz=640, conf=0.0, verbose=False)

    stream = VideoStream(cfg['video_source'], queue_size=cfg.get(
        'prefetch_input_stream_frame', 50)).start()
    db = DBManager(cfg['db_dir'], cfg['max_db_size_mb'])
    sim_checker = SimilarityChecker(db, cfg['similarity_threshold'])
    notifier = Notifier(cfg)

    detection_queue = queue.Queue()
    notify_queue = queue.Queue()

    # Thread 1: Detector
    threading.Thread(target=detection_worker, args=(
        cfg, model, stream, detection_queue, device), daemon=True).start()
    # Thread 2: Similarity Matcher
    threading.Thread(target=similarity_worker, args=(
        cfg, detection_queue, notify_queue, sim_checker), daemon=True).start()
    # Thread 3: UDP Notifier + DB
    threading.Thread(target=notifier_worker, args=(
        cfg, notify_queue, db, notifier), daemon=True).start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")


if __name__ == "__main__":
    main()
