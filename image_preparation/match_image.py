import os
import cv2
import shutil
import glob
import time
import imagehash
from PIL import Image

def compute_hash(image_path):
    try:
        # Use cv2 to load, then PIL to convert to Image object
        img = cv2.imread(image_path)
        if img is None:
            print(f"Failed to load image: {image_path}")
            return None
        # Convert color from BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img)
        return str(imagehash.phash(pil_img))
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def build_old_hashmap(old_dir, img_exts=(".jpg", ".jpeg", ".png")):
    """
    Build a dictionary mapping perceptual hash of images in old_dir 
    to a tuple (image_path, annotation_path). Assumes annotation file
    has same basename with .txt extension.
    """
    hashmap = {}
    for ext in img_exts:
        for img_path in glob.glob(os.path.join(old_dir, f"*{ext}")):
            h = compute_hash(img_path)
            if h is None:
                continue
            # Assume annotation file is in the same directory
            base = os.path.splitext(img_path)[0]
            ann_path = base + ".txt"
            if os.path.exists(ann_path):
                hashmap[h] = (img_path, ann_path)
            else:
                print(f"No corresponding annotation for {img_path}")
    return hashmap

def match_and_copy(old_hashmap, new_dir, output_img_dir, output_ann_dir, img_exts=(".jpg", ".jpeg", ".png")):
    """
    For each image in new_dir, compute hash and if the same hash exists in the old_hashmap,
    copy the new image and the corresponding annotation file from the old dataset
    into the output directories (using the new image filename).
    """
    os.makedirs(output_img_dir, exist_ok=True)
    os.makedirs(output_ann_dir, exist_ok=True)

    unmatched = 0
    matched = 0
    for ext in img_exts:
        for new_img in glob.glob(os.path.join(new_dir, f"*{ext}")):
            h = compute_hash(new_img)
            if h is None:
                continue
            if h in old_hashmap:
                _, old_ann = old_hashmap[h]
                # Define destination paths with the new image's basename
                dest_img = os.path.join(output_img_dir, os.path.basename(new_img))
                dest_ann = os.path.join(output_ann_dir, os.path.splitext(os.path.basename(new_img))[0] + ".txt")
                shutil.copy2(new_img, dest_img)
                shutil.copy2(old_ann, dest_ann)
                matched += 1
                print(f"Matched and copied: {new_img}")
            else:
                unmatched += 1
                print(f"No match for: {new_img}")
    print(f"Total matched: {matched}, Total unmatched: {unmatched}")

def main():
    # Hardcoded directory paths
    old_dir = "/home/kunal/Projects/BSS/EdgeComputing/old_images"
    new_dir = "/home/kunal/Projects/BSS/EdgeComputing/new_images"
    output_img_dir = "/home/kunal/Projects/BSS/EdgeComputing/output_images"
    output_ann_dir = "/home/kunal/Projects/BSS/EdgeComputing/output_annotations"
    
    start = time.time()
    print("Building old image hashmap...")
    old_hashmap = build_old_hashmap(old_dir)
    print(f"Old hashmap built with {len(old_hashmap)} entries")
    
    print("Matching new images with old annotations...")
    match_and_copy(old_hashmap, new_dir, output_img_dir, output_ann_dir)
    
    print(f"Process completed in {time.time()-start:.2f} seconds.")

if __name__ == "__main__":
    main()
