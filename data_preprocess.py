# data_preprocess.py
import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import random

# Config
SRC_DATA_DIR = "datasets/raw"   # expects subfolders per emotion class
OUT_DIR = "datasets/processed"
IMG_SIZE = (224, 224)
TEST_SIZE = 0.1
VAL_SIZE = 0.1
RANDOM_STATE = 42

AUGMENT = True

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def preprocess_image(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    return img

def augment_image(img):
    # Simple augmentation: flip, rotate small angles, brightness
    imgs = [img]
    imgs.append(cv2.flip(img, 1))  # horizontal flip
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), random.uniform(-15, 15), 1)
    imgs.append(cv2.warpAffine(img, M, (cols, rows)))
    # brightness
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    hsv = hsv.astype(np.float32)
    hsv[...,2] = np.clip(hsv[...,2] * random.uniform(0.7, 1.3), 0, 255)
    imgs.append(cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB))
    return imgs

def collect_and_split(src_dir, out_dir):
    ensure_dir(out_dir)
    classes = [d.name for d in Path(src_dir).iterdir() if d.is_dir()]
    print("Found classes:", classes)
    for cls in classes:
        src_cls = Path(src_dir)/cls
        images = [p for p in src_cls.glob("*") if p.suffix.lower() in ['.jpg','.jpeg','.png']]
        train_val, test = train_test_split(images, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        train, val = train_test_split(train_val, test_size=VAL_SIZE/(1-TEST_SIZE), random_state=RANDOM_STATE)
        for split_name, split_list in [("train", train), ("val", val), ("test", test)]:
            out_cls = Path(out_dir)/split_name/cls
            ensure_dir(out_cls)
            for p in split_list:
                img = preprocess_image(p)
                if img is None:
                    continue
                out_path = out_cls / p.name
                cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if AUGMENT and split_name=="train":
                    augmented = augment_image(img)
                    for i, aug in enumerate(augmented[1:]): # skip original (already saved)
                        aug_name = f"{p.stem}_aug{i}{p.suffix}"
                        aug_path = out_cls / aug_name
                        cv2.imwrite(str(aug_path), cv2.cvtColor(aug, cv2.COLOR_RGB2BGR))
    print("Dataset processed at:", out_dir)

if __name__ == "__main__":
    collect_and_split(SRC_DATA_DIR, OUT_DIR)
