# fer2013_to_images.py
import pandas as pd
import numpy as np
import os
import cv2

# Paths
CSV_PATH = "datasets/fer2013.csv"
OUTPUT_DIR = "datasets/raw"

# Emotion labels mapping
emotion_labels = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral"
}

# Create folders
for label in emotion_labels.values():
    os.makedirs(os.path.join(OUTPUT_DIR, label), exist_ok=True)

# Load dataset
data = pd.read_csv(CSV_PATH)
print("Dataset loaded:", data.shape)

for i, row in data.iterrows():
    pixels = np.fromstring(row['pixels'], dtype=int, sep=' ')
    img = pixels.reshape((48,48))
    emotion = emotion_labels[row['emotion']]
    set_type = row['Usage']  # Train / PublicTest / PrivateTest
    
    # Save image
    out_dir = os.path.join(OUTPUT_DIR, emotion)
    filename = f"{emotion}_{i}.jpg"
    cv2.imwrite(os.path.join(out_dir, filename), img)

print("Images saved in datasets/raw/ by emotion class")
