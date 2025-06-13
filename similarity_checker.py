from PIL import Image, ImageDraw, ImageFont
import imagehash
from pathlib import Path
from db_manager import DBManager


class SimilarityChecker:
    def __init__(self, db_manager: DBManager, threshold: float):
        self.db = db_manager
        self.threshold = threshold
        self.mosaic_dir = Path("images/mosaic_results")
        self.mosaic_dir.mkdir(exist_ok=True)

    def is_new(self, crop_img: Image.Image) -> tuple[bool, str | None]:
        hash_val = imagehash.phash(crop_img)
        min_diff = float("inf")

        for item in self.db.hash_value_map:
            try:
                diff = hash_val - self.db.hash_value_map[item] 
                """do we need to calculate the abs difference?"""
                if diff < min_diff:
                    min_diff = diff
                if diff < self.threshold:
                    return False, None
            except Exception as e:
                print(f"Error comparing with {item}: {e}")
                continue

        self.db.hash_value_map[str(hash_val)] = hash_val
        
        return True, str(hash_val)

    def _save_mosaic(self, img1: Image.Image, img2: Image.Image | None, compared_file: str, hash1, hash2, diff, match: bool):
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
