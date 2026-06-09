from tensorflow.keras.models import load_model
import cv2, numpy as np, time
from logger_db import EmotionLogger

MODEL_PATH = "models/emotion_mobilenetv2.h5"
HAAR_PATH = "haarcascades/haarcascade_frontalface_default.xml"
IMG_SIZE = (224,224)

def preprocess_face(face_img):
    face = cv2.resize(face_img, IMG_SIZE)
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype("float32") / 255.0
    return np.expand_dims(face, axis=0)

def main(patient_id="patient_001"):
    logger = EmotionLogger()
    model = load_model(MODEL_PATH)
    face_cascade = logger.load_face_cascade(HAAR_PATH)
    classes = logger.load_classes()

    cap = cv2.VideoCapture(0)
    last_saved_time = 0
    SAVE_INTERVAL = 5

    while True:
        ret, frame = cap.read()
        if not ret: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x,y,w,h) in faces:
            roi = frame[y:y+h, x:x+w]
            face_in = preprocess_face(roi)
            preds = model.predict(face_in, verbose=0)[0]
            idx = int(np.argmax(preds))
            label = classes[idx]
            conf = preds[idx]

            color = (0,255,0) if conf>0.4 else (0,165,255)
            cv2.rectangle(frame, (x,y),(x+w,y+h), color, 2)
            cv2.putText(frame, f"{label} ({conf*100:.1f}%)", (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2)

            now = time.time()
            if now - last_saved_time > SAVE_INTERVAL and conf > 0.3:
                logger.save_emotion(patient_id, label, float(conf))
                last_saved_time = now

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF==ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
