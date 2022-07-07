from glob import glob
from natsort import natsorted
import os
import numpy as np
import cv2

path = "coco/val2014"
file_names = natsorted(glob(os.path.join(path, "*")))

is_exist = os.path.exists(os.path.join(path, "dataset"))
gray_path = os.path.join(path, "dataset/gray/")
rgb_path = os.path.join(path, "dataset/rgb/")
if not is_exist:
    os.mkdir(os.path.join(path, "dataset"))
    os.mkdir(gray_path)
    os.mkdir(rgb_path)

for i in file_names:
    name = i.split(".")[0].split("/")[-1]
    img_bgr = cv2.imread(i)
    img_bgr = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_rgb = img_rgb.astype("uint8")
    img_gray = img_gray.astype("uint8")
    np.save(gray_path + name, img_gray)
    np.save(rgb_path + name, img_rgb)