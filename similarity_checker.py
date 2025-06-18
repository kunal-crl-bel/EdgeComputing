from PIL import Image, ImageDraw, ImageFont
import imagehash
from pathlib import Path
from db_manager import DBManager
import cv2
import numpy as np

class SimilarityChecker:
    def __init__(self, db_manager: DBManager, threshold: float, cfg: dict):
        self.db = db_manager
        self.threshold = threshold
        self.cfg = cfg
        self.mosaic_dir = Path("images/mosaic_results")
        
        # create one SIFT detector for all calls
        self.sift = cv2.SIFT_create()
        # Create BFMatcher with default params (L2 norm for SIFT)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        # how many good matches constitute “the same” image
        self.min_good_matches = cfg.get("min_good_matches", 10)


    def is_new_img_hash(self, crop_img: Image.Image) -> tuple[bool, str | None]:
        hash_val = imagehash.phash(crop_img)

        temp_map = self.db.hash_value_map.copy()  # Avoid modifying while iterating
        for item in temp_map:
            try:
                diff = hash_val - temp_map[item] 
                """do we need to calculate the abs difference?"""
                if diff < self.threshold:
                    del temp_map
                    return False, None
            except Exception as e:
                print(f"Error comparing with {item}: {e}")
                continue
        del temp_map
        
        self.db.hash_value_map[str(hash_val)] = hash_val
        
        if self.cfg.get('mode','') != 'production':
            self.db.add_image(crop_img, id_hash=str(hash_val)) 
            """
                This line is just for Debuging, in realtime,
                we don't need to save the crop image, as the 
                size of cropeed image will be very small, 
                which can be stored in memory also.
            """

        return True, str(hash_val)
    
    def _compute_descriptors(self, pil_img: Image.Image):
        # convert PIL to gray cv2 image
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.sift.detectAndCompute(cv_img, None)
        return descriptors

    def is_new_shift(self, crop_img: Image.Image) -> tuple[bool, str | None]:
        # compute descriptors for the incoming crop
        des = self._compute_descriptors(crop_img)
        if des is None:
            # no features found – treat as new
            new_id = str(len(self.db.descriptor_map))
            self.db.descriptor_map[new_id] = None
            return True, new_id

        # iterate over saved descriptors in your DB
        for stored_id, stored_des in list(self.db.descriptor_map.items()):
            if stored_des is None:
                continue
            # for each stored descriptor set, run knnMatch and apply Lowe’s ratio test
            matches = self.matcher.knnMatch(des, stored_des, k=2)
            good = [m for m,n in matches if m.distance < 0.75 * n.distance]
            if len(good) >= self.min_good_matches:
                # we found enough good matches → it’s not new
                return False, str(stored_id)

        # if we get here, it’s new: save its descriptors
        new_id = str(len(self.db.descriptor_map))
        self.db.descriptor_map[new_id] = des

        # optionally store the image for debugging (as before)
        print(f"New image hash: {new_id}", " mode:", self.cfg.get('mode',''))
        if self.cfg.get('mode','') != 'production':
            self.db.add_image(crop_img, id_hash=new_id)

        return True, new_id
    
    
    def _save_mosaic(self, img1: Image.Image, img2: Image.Image | None, compared_file: str, hash1, hash2, diff, match: bool):
        self.mosaic_dir.mkdir(exist_ok=True)
        
        # Resize both images to same size for display
        img1 = img1.resize((256, 256))
        img2 = img2.resize((256, 256)) if img2 else Image.new("RGB", (256, 256), color="gray")

        # Combine images side by side
        combined = Image.new("RGB", (512, 256))
        combined.paste(img1, (0, 0))
        combined.paste(img2, (256, 0))

        # Draw result text
        draw = ImageDraw.Draw(combined)
        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except:
            font = ImageFont.load_default()

        status = "MATCH" if match else "NO MATCH"
        draw.text((10, 230), f"New Hash: {hash1}", fill="white", font=font)
        if hash2:
            draw.text((10, 245), f"Compared Hash: {hash2} (diff={diff})", fill="white", font=font)
        else:
            draw.text((10, 245), f"Difference={diff})", fill="white", font=font)
            
        draw.text((260, 230), f"Status: {status}", fill="white", font=font)
        draw.text((260, 245), f"Compared: {compared_file}", fill="white", font=font)

        # Save mosaic image
        out_path = self.mosaic_dir / f"mosaic_{hash1}_{'match' if match else 'new'}.jpg"
        combined.save(out_path)
