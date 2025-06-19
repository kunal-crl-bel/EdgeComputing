import imagehash
from PIL import Image

import cv2
import time
st = time.time()
img = cv2.imread('/home/kunal/Projects/BSS/EdgeComputing/images/saved_frames/frame_6.jpg')

img2 = cv2.resize(img, (640*2, 480*2))

# print(img.shape,img2.shape)

h1 = imagehash.phash(Image.fromarray(img))
h2 = imagehash.phash(Image.fromarray(img2))

# print(h1-h2)

print(time.time()-st)