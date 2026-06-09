# app.py
from flask import Flask, render_template, request, redirect, url_for, session, flash, Response, jsonify
import sqlite3, time, cv2, os
from werkzeug.security import generate_password_hash, check_password_hash
from tensorflow.keras.models import load_model
import numpy as np
from flask_socketio import SocketIO, emit
import base64
import tempfile
from datetime import datetime
import librosa
import soundfile as sf
import base64
from logger_db import EmotionLogger
from voice_emotion import predict_voice_emotion
from record_voice import record_voice



app = Flask(__name__)
app.secret_key = "supersecretkey"
socketio = SocketIO(app, cors_allowed_origins="*")

DB_PATH = "emotion_logs.db"
camera = None
is_running = False

# ------------------- Initialize Logger -------------------
logger = EmotionLogger(db_path=DB_PATH)

# ------------------- HOME / INDEX ROUTE -------------------
@app.route("/")
def home():
    # Redirect based on login status and role
    if "email" in session:
        if session["role"] == "therapist":
            return redirect(url_for("therapist_dashboard"))
        elif session["role"] == "patient":
            return redirect(url_for("patient_dashboard"))
    # If not logged in, show landing page
    return render_template("index.html")

# ------------------- AUTH ROUTES -------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        email = request.form["email"]
        name = request.form["name"]
        password = request.form["password"]
        role = request.form["role"]
        password_hash = generate_password_hash(password)

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (email,name,password_hash,role) VALUES (?,?,?,?)",
                      (email, name, password_hash, role))
            conn.commit()
            flash("Account created! Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Email already registered!", "danger")
            return redirect(url_for("register"))
        finally:
            conn.close()
    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT password_hash, role, name FROM users WHERE email=?", (email,))
        row = c.fetchone()
        conn.close()

        if row and check_password_hash(row[0], password):
            session["email"] = email
            session["role"] = row[1]
            session["name"] = row[2]
            if row[1] == "therapist":
                return redirect(url_for("therapist_dashboard"))
            else:
                return redirect(url_for("patient_dashboard"))
        else:
            flash("Invalid email or password", "danger")
    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))

# ------------------- DASHBOARD ROUTES -------------------
@app.route("/therapist/dashboard")
def therapist_dashboard():
    if "email" not in session or session["role"] != "therapist":
        flash("Access denied", "danger")
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT email,name FROM users WHERE role='patient'")
    patients = c.fetchall()
    conn.close()
    return render_template("therapist_dashboard.html", patients=patients)


@app.route("/patient/dashboard")
def patient_dashboard():
    if "email" not in session or session["role"] != "patient":
        flash("Access denied", "danger")
        return redirect(url_for("login"))

    logger = EmotionLogger(db_path=DB_PATH)
    sessions = logger.get_sessions_for_patient(session["email"])
    return render_template("patient_dashboard.html", sessions=sessions)

# ------------------- SESSION BOOKING -------------------
@app.route("/book_session", methods=["GET", "POST"])
def book_session():
    if "email" not in session or session["role"] != "patient":
        flash("Only patients can book sessions.", "danger")
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT email,name FROM users WHERE role='therapist'")
    therapists = c.fetchall()
    conn.close()

    if request.method == "POST":
        therapist_id = request.form["therapist"]
        session_time = request.form["session_time"]
        notes = request.form.get("notes", "")
        logger.book_session(session["email"], therapist_id, session_time, notes)
        flash("Session booked successfully!", "success")
        return redirect(url_for("patient_dashboard"))

    return render_template("book_session.html", therapists=therapists)


@app.route("/manage_sessions")
def manage_sessions():
    if "email" not in session or session["role"] != "therapist":
        flash("Access denied", "danger")
        return redirect(url_for("login"))

    sessions = logger.get_sessions_for_therapist(session["email"])
    return render_template("manage_sessions.html", sessions=sessions)


@app.route("/delete_session/<int:session_id>")
def delete_session(session_id):
    if "email" not in session or session["role"] != "therapist":
        flash("Access denied", "danger")
        return redirect(url_for("login"))
    logger.delete_session(session_id)
    flash("Session deleted!", "success")
    return redirect(url_for("manage_sessions"))

# ------------------- EMOTION MONITORING -------------------
def gen_frames(patient_id):
    global camera, is_running
    model = load_model("models/emotion_mobilenetv2.h5")
    face_cascade = logger.load_face_cascade("haarcascades/haarcascade_frontalface_default.xml")
    classes = logger.load_classes()
    last_saved_time = 0
    SAVE_INTERVAL = 5

    while is_running and camera.isOpened():
        ret, frame = camera.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.3,
    minNeighbors=3,
    minSize=(30, 30)
    )


        for (x, y, w, h) in faces:
            roi = frame[y:y + h, x:x + w]
            face_in = cv2.resize(roi, (224, 224))
            face_in = cv2.cvtColor(face_in, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
            face_in = np.expand_dims(face_in, axis=0)
            preds = model.predict(face_in, verbose=0)[0]
            idx = int(np.argmax(preds))
            label = classes[idx]
            conf = preds[idx]

            color = (0, 255, 0) if conf > 0.2 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"{label} ({conf * 100:.1f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            now = time.time()
            if now - last_saved_time > SAVE_INTERVAL and conf > 0.2:
                logger.save_emotion(patient_id, label, float(conf))
                last_saved_time = now

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route("/start_session/<patient_id>")
def start_session(patient_id):
    global camera, is_running
    if "email" not in session or session["role"] != "therapist":
        flash("Access denied", "danger")
        return redirect(url_for("therapist_dashboard"))

    session["current_patient"] = patient_id
    if not is_running:
        camera = cv2.VideoCapture(0)
        is_running = True
    flash(f"Monitoring started for {patient_id}", "success")
    return redirect(url_for("therapist_dashboard"))


@app.route("/stop_session")
def stop_session():
    global camera, is_running
    if is_running:
        is_running = False
        camera.release()
        camera = None
    session.pop("current_patient", None)
    flash("Session stopped", "info")
    return redirect(url_for("therapist_dashboard"))


@app.route("/video_feed")
def video_feed():
    if "current_patient" not in session:
        flash("No patient selected", "danger")
        return redirect(url_for("therapist_dashboard"))
    return Response(gen_frames(session["current_patient"]),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# ------------------- VIEW EMOTION LOGS -------------------
@app.route("/view_emotions/<patient_id>")
def view_emotions(patient_id):

    if "email" not in session:
        return redirect(url_for("login"))

    if session["role"] == "patient" and session["email"] != patient_id:
        flash("Access denied", "danger")
        return redirect(url_for("patient_dashboard"))

    raw_logs = logger.get_emotions(patient_id)

    formatted_logs = []
    for log in raw_logs:
        emotion_type, emotion, confidence, voice_emotion, voice_confidence, timestamp = log

        if timestamp:
            try:
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except:
                pass

        formatted_logs.append(
            (emotion_type, emotion, confidence,
             voice_emotion, voice_confidence,
             timestamp)
        )

    return render_template("view_emotions.html",
                           logs=formatted_logs,
                           patient_id=patient_id)


@app.route("/delete_patient/<string:email>")
def delete_patient(email):
    if "email" not in session or session["role"] != "therapist":
        flash("Access denied", "danger")
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("DELETE FROM users WHERE email=? AND role='patient'", (email,))
        c.execute("DELETE FROM emotions WHERE patient_id=?", (email,))
        conn.commit()
        flash(f"Patient {email} deleted successfully!", "success")
    except Exception as e:
        flash(f"Error deleting patient: {e}", "danger")
    finally:
        conn.close()

    return redirect(url_for("therapist_dashboard"))

# ------------------- VOICE EMOTION (THERAPIST ONLY) -------------------

@app.route("/record_voice/<patient_id>")
def record_voice_emotion(patient_id):

    if "email" not in session or session["role"] != "therapist":
        flash("Access denied", "danger")
        return redirect(url_for("login"))

    try:
        print("Voice route triggered")
        print("Starting recording...")

        record_voice("voice_input.wav")

        print("Recording finished")

        emotion, confidence = predict_voice_emotion("voice_input.wav")

        print("Prediction:", emotion, confidence)

        logger.save_voice_emotion(patient_id, emotion, confidence)

        flash(f"Voice emotion '{emotion}' recorded successfully!", "success")

    except Exception as e:
        print("ERROR:", e)
        flash(f"Voice emotion error: {e}", "danger")

    return redirect(url_for("therapist_dashboard"))

@app.route("/upload_voice", methods=["POST"])
def upload_voice():

    if "email" not in session or session["role"] != "therapist":
        return jsonify({"error": "Access denied"}), 403

    audio = request.files["audio"]
    patient_id = request.form["patient_id"]

    filename = "temp_voice.wav"
    audio.save(filename)

    emotion, confidence = predict_voice_emotion(filename)

    logger.save_voice_emotion(patient_id, emotion, confidence)

    os.remove(filename)

    return jsonify({
        "emotion": emotion,
        "confidence": confidence
    })

@app.route("/voice_record/<patient_id>")
def voice_record_page(patient_id):

    if "email" not in session or session["role"] != "therapist":
        flash("Access denied", "danger")
        return redirect(url_for("login"))

    return render_template("voice_record.html", patient_id=patient_id)

@socketio.on("audio_chunk")
def handle_audio(data):

    try:
        audio_bytes = base64.b64decode(data["audio"])
        patient_id = data["patient_id"]

        temp_webm = "temp_chunk.webm"
        temp_wav = "temp_chunk.wav"

        # Save webm
        with open(temp_webm, "wb") as f:
            f.write(audio_bytes)

        # Convert webm to wav using librosa
        audio_bytes = base64.b64decode(data["audio"])
        temp_wav = "temp_stream.wav"

        with open(temp_wav, "wb") as f:
            f.write(audio_bytes)

        # Load properly using librosa
        y, sr = librosa.load(temp_wav, sr=22050)

        # Rewrite clean WAV file
        sf.write(temp_wav, y, sr)

        emotion, confidence = predict_voice_emotion(temp_wav)

        logger.save_voice_emotion(patient_id, emotion, confidence)

        os.remove(temp_webm)
        os.remove(temp_wav)

        emit("emotion_result", {
            "emotion": emotion,
            "confidence": confidence
        })

    except Exception as e:
        print("VOICE STREAM ERROR:", e)
        emit("emotion_result", {
            "emotion": "Silence",
            "confidence": 0.0
        })


# ------------------- RUN APP -------------------
if __name__ == "__main__":
    socketio.run(app, debug=True)

