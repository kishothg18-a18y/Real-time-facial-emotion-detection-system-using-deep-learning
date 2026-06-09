import os

DATA_DIR = "datasets/raw"
for cls in os.listdir(DATA_DIR):
    cls_path = os.path.join(DATA_DIR, cls)
    if os.path.isdir(cls_path):
        print(cls, "->", len(os.listdir(cls_path)), "images")
