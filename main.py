import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from config import load_config
from db_manager import DBManager
from similarity_checker import SimilarityChecker
from notifier import Notifier
from VideoStream import VideoStream


prev_tacked_map = dict()

def main() -> None:
    global prev_tacked_map
    
    
    print("Started")
    cfg = load_config("config.xml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare directories
    saved_dir = cfg.get('saved_frame_dir', 'saved_frames')
    os.makedirs(saved_dir, exist_ok=True)

    # Load and optimize model once
    model = YOLO(cfg['model_path'])
    model.to(device)
    model.fuse()  # speed up inference
    model.model.half()  # use FP16 if supported

    # Model heat-up: run a dummy inference to initialize CUDA kernels
    print("Warming up model...")
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    with torch.amp.autocast('cuda',enabled=True):
        _ = model.predict(source=dummy, device=device, imgsz=640, conf=0.0,verbose=False)
    print("Model heat-up complete")

    # Prefetch video frames
    try:
        stream = VideoStream(cfg['video_source'], queue_size=cfg.get('prefetch_input_stream_frame', 50)).start()
    except Exception as e:
        print("Error iin reading Video stream is: ", e)
    
    
    if stream:
        print("Started reading stream.")
        
    # Executors for similarity checks and disk writes
    try:
        sim_executor = ThreadPoolExecutor(max_workers=cfg['parallel_workers_similarity'])
    except Exception as e:
        print("Error in sim_executor: ",e)
    
    try:
        io_executor = ThreadPoolExecutor(max_workers=cfg['parallel_workers_io'])
    except Exception as e:
        print("Error in io_executor: ",e)

    db = DBManager(cfg['db_dir'], cfg['max_db_size_mb'])
    sim = SimilarityChecker(db, cfg['similarity_threshold'])
    notifier = Notifier(cfg)

    frame_idx = 0
    try:
        while stream.more():
            frame = stream.read()
            if frame is None:
                break

            frame_idx += 1
            print(f"Processing frame {frame_idx}")

            # Task 1: detect + track with autocast
            with torch.amp.autocast('cuda',enabled=True):
                tracks = model.track(
                    source=frame,
                    device=device,
                    imgsz=640,
                    conf=float(cfg['detect_confidence']),
                    tracker="bytetrack.yaml",
                    verbose=False,
                )[0]

            # Task 2: similarity checks in parallel
            futures = []
            for box in tracks.boxes:
                tid = int(box.id)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                fut = sim_executor.submit(sim.is_new, pil_img, tid)
                fut.track_id = tid
                fut.img = pil_img
                fut.class_id = int(box.cls)  # for SRS lookup
                futures.append(fut)

            # Task 3: handle new events
            counter_map = {}
            for fut in as_completed(futures):
                is_new, hashValue, id = fut.result()
                
                if id in prev_tacked_map:
                    print("id is : ",id)
                    continue  # Skip if already processed
                else:
                    print("id not is : ",id)
                prev_tacked_map[id] = True  # Mark this ID as processed
                    
                if not is_new :
                    print(f"  → Track {fut.track_id} is not new, skipping")
                    continue
                
                
                # Lookup SRS for detected class and count
                srs_val = cfg['srs_values_of_classes'].get(model.names.get(fut.class_id,None), 0)
                if counter_map.get(srs_val,None):
                    counter_map[srs_val] += counter_map.get(srs_val, 1)
                else:
                    counter_map[srs_val] = 1
                    
                
                # Log image crop + hashValue
                db.add_image(fut.img, hashValue)
        
                    
            for id in counter_map:
                notifier.add(id,counter_map[id])
                
            if len(counter_map)>0:
                filename = f"frame_{frame_idx:06d}.jpg"
                path = os.path.join(saved_dir, filename)
                overlay = tracks.plot()
                io_executor.submit(cv2.imwrite, path, overlay)
                print(f"  → scheduled save new-track frame to {path}")
                    
            

    finally:
        stream.stop()
        sim_executor.shutdown(wait=True)
        io_executor.shutdown(wait=True)
        notifier.close()

if __name__ == '__main__':
    main()
