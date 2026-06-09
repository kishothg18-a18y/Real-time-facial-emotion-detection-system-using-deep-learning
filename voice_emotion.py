# voice_emotion.py

import numpy as np
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import os

MODEL_PATH = "models/voice_emotion_model.h5"
SAMPLE_RATE = 22050
N_MFCC = 40

# Same emotions used in training
EMOTIONS = [
    "neutral", "calm", "happy", "sad",
    "angry", "fear", "disgust", "surprise"
]

# Load model once
model = load_model(MODEL_PATH)

# Encode labels same as training
le = LabelEncoder()
le.fit(EMOTIONS)


def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

    if len(audio) < 2048:
     raise ValueError("Audio too short for prediction")

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc


def predict_voice_emotion(file_path):
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=0)

    predictions = model.predict(features, verbose=0)
    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions))

    emotion = le.inverse_transform([predicted_index])[0]

    return emotion, confidence
