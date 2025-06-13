import os
import cv2
import torch
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

from config import load_config
from db_manager import DBManager
from similarity_checker import SimilarityChecker
from notifier import Notifier

def main() -> None:
    print("Started")
    cfg = load_config("config.xml")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    saved_dir = cfg.get('saved_frame_dir', 'saved_frames')
    os.makedirs(saved_dir, exist_ok=True)

    model = YOLO(cfg['model_path'])
    cap = cv2.VideoCapture(cfg['video_source'])
    executor = ThreadPoolExecutor(max_workers=cfg['parallel_workers'])
    db = DBManager(cfg['db_dir'], cfg['max_db_size_mb'])
    sim = SimilarityChecker(db, cfg['similarity_threshold'])
    notifier = Notifier(cfg)

    frame_idx = 0
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {cfg['video_source']}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            # detect + track
            tracks = model.track(
                source=frame,
                device=device,
                conf=float(cfg['detect_confidence']),
                tracker="bytetrack.yaml",
                verbose=False,
            )[0]

            futures = []
            for box in tracks.boxes:
                tid = int(box.id)
                x1, y1, x2, y2 = box.xyxy.tolist()[0]
                crop = frame[int(y1):int(y2), int(x1):int(x2)]
                if crop.size == 0:
                    continue
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                fut = executor.submit(sim.is_new, pil_img)
                fut.track_id = tid
                fut.img = pil_img
                futures.append(fut)

            for fut in as_completed(futures):
                is_new, hist = fut.result()
                if not is_new:
                    continue

                # 1) Notify + DB as before
                notifier.add(fut.track_id)
                """
                I have some SRS values of corresponding class, 
                that i detected, so in my message i need to send,
                that SRC value for object and the count.
                """
                db.add_image(fut.img, hist)

                # 2) SAVE THE FULL FRAME
                # e.g. saved_frames/frame_123_track7.jpg
                filename = f"frame_{frame_idx:06d}_track{fut.track_id}.jpg"
                path = os.path.join(saved_dir, filename)
                frame = tracks.plot()
                
                cv2.imwrite(path, frame)
                print(f"  → saved new-track frame to {path}")

    finally:
        cap.release()
        executor.shutdown()
        notifier.close()

if __name__ == '__main__':
    main()
