import cv2
import face_recognition
import numpy as np
from picamera2 import Picamera2
import time
import pickle
import datetime
import threading
import os
import sqlite3
import smtplib
from email.message import EmailMessage
import imghdr

# ---------------- CONFIGURACION CORREO GMAIL ----------------
EMAIL_SENDER = "latostada18@gmail.com"
EMAIL_PASSWORD = "prfe xusb oqgz layp"

EMAIL_RECEIVERS = [
    "caam314@gmail.com"
]

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
# ------------------------------------------------------------

# Carpetas para guardar imagenes
KNOWN_IMG_FOLDER = "capturas_conocidos"
UNKNOWN_IMG_FOLDER = "capturas_desconocidos"
os.makedirs(KNOWN_IMG_FOLDER, exist_ok=True)
os.makedirs(UNKNOWN_IMG_FOLDER, exist_ok=True)

DB_NAME = "log_eventos.db"
faces_seen_in_last_frame = set()

def setup_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS eventos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        nombre_detectado TEXT NOT NULL,
        tipo_evento TEXT NOT NULL,
        nombre_clip TEXT NOT NULL,
        alerta_enviada INTEGER NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def log_event_to_db(timestamp, name, event_type, filename, alert_sent):
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        sql = """
        INSERT INTO eventos (timestamp, nombre_detectado, tipo_evento, nombre_clip, alerta_enviada)
        VALUES (?, ?, ?, ?, ?)
        """
        cursor.execute(sql, (timestamp, name, event_type, filename, alert_sent))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR] DB: {e}")

def send_email_alert(filepath):
    try:
        msg = EmailMessage()
        msg["Subject"] = "ALERTA: Rostro desconocido detectado"
        msg["From"] = EMAIL_SENDER
        msg["To"] = ", ".join(EMAIL_RECEIVERS)
        msg.set_content("Se ha detectado un rostro desconocido. Imagen adjunta.")

        with open(filepath, "rb") as f:
            img_data = f.read()
            img_type = imghdr.what(None, img_data) or "jpeg"
            msg.add_attachment(img_data, maintype="image", subtype=img_type, filename=os.path.basename(filepath))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as smtp:
            smtp.starttls()
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)

        print(f"[CORREO] Enviado: {filepath}")

    except Exception as e:
        print(f"[ERROR] No se pudo enviar correo: {e}")

def save_face_capture(name, frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    if name == "Unknown":
        filename = f"unknown_{timestamp}.jpg"
        filepath = os.path.join(UNKNOWN_IMG_FOLDER, filename)
        cv2.imwrite(filepath, frame)

        threading.Thread(target=send_email_alert, args=(filepath,)).start()
        log_event_to_db(timestamp, name, "Desconocido", filename, 1)
        print(f"[ALERTA] Desconocido guardado y correo enviado: {filepath}")
    else:
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(KNOWN_IMG_FOLDER, filename)
        cv2.imwrite(filepath, frame)
        log_event_to_db(timestamp, name, "Conocido", filename, 0)
        print(f"[INFO] Conocido guardado: {filepath}")

print("[INFO] Cargando encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (1920, 1080)}))
picam2.start()

setup_database()

cv_scaler = 4
frame_count = 0
start_time = time.time()
fps = 0

RECOGNIZE_EVERY_N_FRAMES = 4
last_face_locations = []
last_face_names = []

while True:
    frame = picam2.capture_array()

    small = cv2.resize(frame, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
    rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    frame_count += 1
    do_recognition = (frame_count % RECOGNIZE_EVERY_N_FRAMES == 0)

    if do_recognition:
        face_locations = face_recognition.face_locations(rgb_small, model='hog')
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations, model='small')
        face_names = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.48)
            name = "Unknown"

            if known_face_encodings:
                face_dist = face_recognition.face_distance(known_face_encodings, face_encoding)
                best = np.argmin(face_dist)
                if matches and matches[best]:
                    name = known_face_names[best]

            face_names.append(name)

        current_faces = set(face_names)
        new_faces = current_faces - faces_seen_in_last_frame
        for name in new_faces:
            save_face_capture(name, frame)
        faces_seen_in_last_frame = current_faces

        last_face_locations = face_locations
        last_face_names = face_names

    for (top, right, bottom, left), name in zip(last_face_locations, last_face_names):
        top *= cv_scaler; right *= cv_scaler; bottom *= cv_scaler; left *= cv_scaler
        color = (3, 42, 244) if name == "Unknown" else (244, 42, 3)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 3)
        cv2.rectangle(frame, (left - 3, top - 35), (right + 3, top), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    elapsed_time = time.time() - start_time
    if elapsed_time > 1:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Video", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cv2.destroyAllWindows()
picam2.stop()
print("[INFO] Aplicacion cerrada.")
