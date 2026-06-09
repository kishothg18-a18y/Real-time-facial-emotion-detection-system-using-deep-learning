import sqlite3

DB_PATH = "emotion_logs.db"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

try:
    c.execute("ALTER TABLE emotions ADD COLUMN emotion_type TEXT")
    print("✔ emotion_type column added")
except sqlite3.OperationalError as e:
    print("emotion_type:", e)

try:
    c.execute("ALTER TABLE emotions ADD COLUMN voice_emotion TEXT")
    print("✔ voice_emotion column added")
except sqlite3.OperationalError as e:
    print("voice_emotion:", e)

try:
    c.execute("ALTER TABLE emotions ADD COLUMN voice_confidence REAL")
    print("✔ voice_confidence column added")
except sqlite3.OperationalError as e:
    print("voice_confidence:", e)

conn.commit()
conn.close()
