# logger_db.py
import sqlite3
import os
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from datetime import datetime


DB_PATH = "emotion_logs.db"
MODEL_PATH = "models/emotion_mobilenetv2.h5"
HAAR_PATH = "haarcascades/haarcascade_frontalface_default.xml"
IMG_SIZE = (224, 224)

class EmotionLogger:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.classes = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
        self._create_tables()

    # ---------------- Database Setup ----------------
    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _create_tables(self):
        conn = self._connect()
        c = conn.cursor()
        # Users table
        c.execute('''CREATE TABLE IF NOT EXISTS users(
                     email TEXT PRIMARY KEY, name TEXT, password_hash TEXT, role TEXT)''')
        # Emotion logs
        c.execute('''CREATE TABLE IF NOT EXISTS emotions(
             id INTEGER PRIMARY KEY AUTOINCREMENT,
             patient_id TEXT,
             emotion_type TEXT,
             emotion TEXT,
             confidence REAL,
             voice_emotion TEXT,
             voice_confidence REAL,
             timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

        # Therapy sessions
        c.execute('''CREATE TABLE IF NOT EXISTS sessions(
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     patient_id TEXT,
                     therapist_id TEXT,
                     session_time TEXT,
                     notes TEXT)''')
        conn.commit()
        conn.close()

    # ----------------- Helper Methods ----------------
    
    def save_emotion(self, patient_id, emotion, confidence):
        conn = self._connect()
        c = conn.cursor()

        local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        c.execute("""
            INSERT INTO emotions 
            (patient_id, emotion, confidence, emotion_type, timestamp)
            VALUES (?, ?, ?, 'FACE', ?)
        """, (patient_id, emotion, confidence, local_time))

        conn.commit()
        conn.close()

    def save_voice_emotion(self, patient_id, voice_emotion, voice_confidence):
        conn = self._connect()
        c = conn.cursor()

        local_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        c.execute("""
            INSERT INTO emotions
            (patient_id, voice_emotion, voice_confidence, emotion_type, timestamp)
            VALUES (?, ?, ?, 'VOICE', ?)
        """, (patient_id, voice_emotion, voice_confidence, local_time))

        conn.commit()
        conn.close()


    def get_emotions(self, patient_id):
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            SELECT emotion_type, emotion, confidence,
                voice_emotion, voice_confidence, timestamp
            FROM emotions
            WHERE patient_id=?
            ORDER BY timestamp DESC
        """, (patient_id,))
        rows = c.fetchall()
        conn.close()
        return rows


    # ----------------- Session Booking ----------------
    def book_session(self, patient_id, therapist_id, session_time, notes=""):
        conn = self._connect()
        c = conn.cursor()
        c.execute("INSERT INTO sessions(patient_id, therapist_id, session_time, notes) VALUES (?, ?, ?, ?)",
                  (patient_id, therapist_id, session_time, notes))
        conn.commit()
        conn.close()

    def get_sessions_for_therapist(self, therapist_id):
        conn = self._connect()
        c = conn.cursor()
        c.execute("SELECT id, patient_id, therapist_id, session_time, notes FROM sessions WHERE therapist_id=? ORDER BY session_time", (therapist_id,))
        rows = c.fetchall()
        conn.close()
        return rows

    def get_sessions_for_patient(self, patient_email):
        conn = self._connect()
        c = conn.cursor()
        c.execute("""
            SELECT therapist_id, session_time, notes
            FROM sessions
            WHERE patient_id=?
            ORDER BY session_time ASC
        """, (patient_email,))
        rows = c.fetchall()
        conn.close()

        # Convert therapist_id to therapist name
        sessions = []
        for therapist_id, session_time, notes in rows:
            conn = self._connect()
            c = conn.cursor()
            c.execute("SELECT name FROM users WHERE email=?", (therapist_id,))
            therapist_name = c.fetchone()
            therapist_name = therapist_name[0] if therapist_name else therapist_id
            conn.close()
            sessions.append({"therapist_name": therapist_name, "session_time": session_time, "notes": notes})
        return sessions

    def delete_session(self, session_id):
        conn = self._connect()
        c = conn.cursor()
        c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        conn.commit()
        conn.close()

    # ----------------- Model Helpers ----------------
    def load_model_for_dashboard(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        return load_model(MODEL_PATH)

    def load_face_cascade(self, haar_path=HAAR_PATH):
        if os.path.exists(haar_path):
            face_cascade = cv2.CascadeClassifier(haar_path)
        else:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if face_cascade.empty():
            raise IOError("Could not load HaarCascade classifier")
        return face_cascade

    def preprocess_face(self, face_img):
        face = cv2.resize(face_img, IMG_SIZE)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
        return np.expand_dims(face, axis=0)

    def load_classes(self, train_dir="datasets/processed/train"):
        if os.path.exists(train_dir):
            classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            return classes if classes else self.classes
        return self.classes
