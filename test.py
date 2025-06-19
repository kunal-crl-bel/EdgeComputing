from ultralytics.trackers.byte_tracker import BYTETracker

import torch
import time

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
    global frame_map, reported_ids, last_seen

    # Instantiate the BYTETracker tracker once
    tracker = BYTETracker(
        tracker_config=cfg.get('tracker_config', 'bytetrack.yaml'),
        device=device
    )

    while stream.more():
        frame = stream.read()
        if frame is None:
            continue

        # 1) Run *detection only* (no internal tracker)
        with torch.amp.autocast(device_type='cuda', enabled=(device=='cuda')):
            det_results = model.predict(
                source=frame,
                device=device,
                imgsz=int(cfg.get('imgsz', 640)),
                conf=float(cfg.get('detect_confidence', 0.25)),
                verbose=False,
                # ensure we only get raw boxes, not tracks
                tracker=None
            )[0]

        # 2) Convert raw detections to ByteTrack input format
        #    each row: [x1, y1, x2, y2, confidence, class_id]
        dets = []
        for box in det_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls  = int(box.cls[0])
            dets.append([x1, y1, x2, y2, conf, cls])
        dets = torch.tensor(dets, device=device)

        # 3) Feed into tracker
        #    tracker.update returns a list of track objects
        tracks = tracker.update(dets, frame.shape[:2], frame)

        # 4) Annotate frame and store it
        annotated_frame = det_results.plot()  # or you could draw tracks manually
        frame_map[frame_id] = annotated_frame

        # 5) Process each active track exactly as before
        h, w = frame.shape[:2]
        for trk in tracks:
            tid      = trk.track_id
            class_id = trk.cls  # same as your class mapping
            bbox     = trk.xyxy  # [x1, y1, x2, y2]

            if class_id not in CLASS_ID_TO_SRS:
                continue
            if tid in reported_ids:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            crop = frame[y1:y2, x1:x2]
            detection_queue.put((tid, crop, class_id, frame_id))
            last_seen[tid] = time.time()

        frame_id += 1

    print("[Detection] Finished")
