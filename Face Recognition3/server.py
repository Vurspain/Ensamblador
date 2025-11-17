import subprocess
import threading
import os
import sys
import face_recognition
import pickle
import numpy as np
import datetime
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from picamera2 import Picamera2
import time
from flask import Flask, render_template, Response, request, jsonify
import cv2
import sqlite3

app = Flask(__name__)

ENCODINGS_FILE = "encodings.pickle"
DB_NAME = "log_eventos.db"

# --- CONFIGURACIÓN DE VARIABLES ---
EMAIL_SENDER = "latostada18@gmail.com"
EMAIL_PASSWORD = "prfe xusb oqgz layp"
EMAIL_RECEIVERS = ["caam314@gmail.com"] 

KNOWN_FACES_DIR = "capturas_conocidos"
UNKNOWN_FACES_DIR = "capturas_desconocidos"
DETECTION_COOLDOWN = 10 
EMAIL_COOLDOWN = 60 * 5 

# Estados del sistema
CURRENT_MODE = "inactive"
CURRENT_STATUS_MSG = ""
CAPTURE_NAME = ""
CAPTURE_COUNT = 0
MAX_CAPTURES = 10
LAST_CAPTURE_TIME = 0

# Listas de control
authorized_names = ["carlos", "valeria", "marcos", "anali", "ian"]
last_detection_time = {}
last_email_sent_time = {}
face_locations = []
face_names = []
cv_scaler = 4 

# Asegurar carpetas
if not os.path.exists(KNOWN_FACES_DIR): os.makedirs(KNOWN_FACES_DIR)
if not os.path.exists(UNKNOWN_FACES_DIR): os.makedirs(UNKNOWN_FACES_DIR)

# --- BASE DE DATOS ---
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
    print("[INFO] Base de datos verificada.")

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
        print(f"[DB] Evento registrado: {name} - {event_type}")
    except Exception as e:
        print(f"[ERROR] DB: {e}")

def cargar_nombres():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.loads(f.read())
            return list(set(data["names"]))
    return []

# --- INICIO DE CÁMARA Y MODELO ---
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
time.sleep(1.0)

print("[INFO] Cargando encodings...")
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# --- FUNCIONES AUXILIARES ---
def send_email_alert(name, image_path):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = ", ".join(EMAIL_RECEIVERS)
        
        if name == "Unknown":
            msg['Subject'] = "Alerta: Persona Desconocida Detectada"
            body = "Se ha detectado a una persona desconocida."
        else:
            msg['Subject'] = f"Alerta: {name} Detectado (No Autorizado)"
            body = f"Se ha detectado a {name}, quien no está en la lista de autorizados."
        
        msg.attach(MIMEText(body, 'plain'))
        attachment = open(image_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % os.path.basename(image_path))
        msg.attach(part)
        
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVERS, text)
        server.quit()
        print(f"[INFO] Email enviado para {name}.")
    except Exception as e:
        print(f"[ERROR] Email: {e}")

def save_capture(frame_bgr, name):
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    timestamp_db = now.strftime("%Y-%m-%d %H:%M:%S") # Formato limpio para DB
    
    if name == "Unknown":
        folder = UNKNOWN_FACES_DIR
        filename = f"unknown_{timestamp}.jpg"
        event_type = "Desconocido"
    else:
        folder = KNOWN_FACES_DIR
        filename = f"{name.lower()}_{timestamp}.jpg"
        event_type = "Conocido"
        
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame_bgr)
    
    # IMPORTANTE: Aquí devolvemos más datos para poder registrar en DB
    return filepath, filename, timestamp_db, event_type

def generar_video():
    global face_locations, face_names, last_detection_time, last_email_sent_time
    global CURRENT_MODE, CURRENT_STATUS_MSG, CAPTURE_NAME, CAPTURE_COUNT, LAST_CAPTURE_TIME
    global known_face_encodings, known_face_names

    while True:
        frame_xrgb = picam2.capture_array()
        # CORREGIDO: Usamos BGRA2BGR para evitar colores invertidos (azul vs rojo)
        frame_bgr = cv2.cvtColor(frame_xrgb, cv2.COLOR_BGRA2BGR)
        display_frame = frame_bgr.copy()
        current_time = time.time()
        
        # === MODO DETECCIÓN ===
        if CURRENT_MODE == "detection":
            resized_frame = cv2.resize(frame_bgr, (0, 0), fx=(1/cv_scaler), fy=(1/cv_scaler))
            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                face_names.append(name)
                
                # --- Lógica de Evento y Alerta ---
                if current_time - last_detection_time.get(name, 0) > DETECTION_COOLDOWN:
                    last_detection_time[name] = current_time
                    
                    # 1. Guardar foto y obtener datos
                    image_path, filename, timestamp_db, event_type = save_capture(frame_bgr, name)
                    
                    alert_sent = 0
                    # 2. Enviar Email si corresponde
                    if (name == "Unknown" or name not in authorized_names) and (current_time - last_email_sent_time.get(name, 0) > EMAIL_COOLDOWN):
                        last_email_sent_time[name] = current_time
                        alert_sent = 1
                        email_thread = threading.Thread(target=send_email_alert, args=(name, image_path))
                        email_thread.start()
                    
                    # 3. REGISTRAR EN BASE DE DATOS (¡NUEVO!)
                    # Usamos un hilo para no frenar el video con la escritura en disco/db
                    db_thread = threading.Thread(target=log_event_to_db, args=(timestamp_db, name, event_type, filename, alert_sent))
                    db_thread.start()

            # Dibujar
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= cv_scaler; right *= cv_scaler; bottom *= cv_scaler; left *= cv_scaler
                # Color rojo si es desconocido o no autorizado, verde si es autorizado
                color = (0, 255, 0) if name in authorized_names else (0, 0, 255) 
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 3)
                cv2.rectangle(display_frame, (left -3, top - 35), (right+3, top), color, cv2.FILLED)
                cv2.putText(display_frame, name, (left + 6, top - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)
                
                if name in authorized_names:
                    cv2.putText(display_frame, "AUTORIZADO", (left + 6, bottom + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1)
                else:
                    cv2.putText(display_frame, "NO AUTORIZADO", (left + 6, bottom + 23), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 255), 1)

        # === MODO CAPTURA ===
        elif CURRENT_MODE == "capture":
            if (current_time - LAST_CAPTURE_TIME > 1): 
                LAST_CAPTURE_TIME = current_time
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{CAPTURE_NAME}_{timestamp}.jpg"
                filepath = os.path.join("dataset", CAPTURE_NAME, filename)
                cv2.imwrite(filepath, frame_bgr)
                
                CAPTURE_COUNT += 1
                CURRENT_STATUS_MSG = f"Capturando foto {CAPTURE_COUNT}/{MAX_CAPTURES}..."
                print(f"Foto {CAPTURE_COUNT}/{MAX_CAPTURES} guardada para {CAPTURE_NAME}")

            cv2.putText(display_frame, f"CAPTURANDO: {CAPTURE_COUNT}/{MAX_CAPTURES}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            if CAPTURE_COUNT >= MAX_CAPTURES:
                CURRENT_MODE = "training"
                CURRENT_STATUS_MSG = "Captura finalizada. Re-entrenando..."
                
                def train_model():
                    global CURRENT_MODE, CURRENT_STATUS_MSG, known_face_encodings, known_face_names
                    try:
                        subprocess.run(["python", "model_training.py"], check=True)
                        with open("encodings.pickle", "rb") as f:
                            data = pickle.loads(f.read())
                        known_face_encodings = data["encodings"]
                        known_face_names = data["names"]
                        CURRENT_STATUS_MSG = "¡Entrenamiento completado!"
                        time.sleep(5)
                        CURRENT_STATUS_MSG = ""
                        CURRENT_MODE = "detection"
                    except Exception as e:
                        print(f"Error: {e}")
                        CURRENT_STATUS_MSG = "Error entrenamiento."
                        time.sleep(5)
                        CURRENT_MODE = "detection"
                
                threading.Thread(target=train_model).start()
        
        elif CURRENT_MODE == "training":
            cv2.putText(display_frame, "ENTRENANDO...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret: continue
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# --- RUTAS FLASK ---

@app.route('/')
def dashboard():
    return render_template("dashboard.html")

@app.route('/video')
def video():
    return Response(generar_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode/<string:mode_name>')
def set_mode(mode_name):
    global CURRENT_MODE, face_locations, face_names
    if mode_name == "detection":
        CURRENT_MODE = "detection"
    elif mode_name == "inactive":
        CURRENT_MODE = "inactive"
        face_locations = [] 
        face_names = []
    return jsonify({"new_mode": CURRENT_MODE})

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global CURRENT_MODE, CAPTURE_NAME, CAPTURE_COUNT, CURRENT_STATUS_MSG
    name = request.form.get("name")
    if not name: return jsonify({"status": "Error"}), 400

    CAPTURE_NAME = name
    CAPTURE_COUNT = 0
    CURRENT_MODE = "capture"
    CURRENT_STATUS_MSG = f"Iniciando captura para {CAPTURE_NAME}..."
    
    dataset_folder = os.path.join("dataset", name)
    if not os.path.exists(dataset_folder): os.makedirs(dataset_folder)
        
    return jsonify({"status": "ok"})

@app.route('/get_status')
def get_status():
    global CURRENT_MODE, CURRENT_STATUS_MSG
    return jsonify({"mode": CURRENT_MODE, "message": CURRENT_STATUS_MSG})

# --- NUEVA RUTA API PARA LA TABLA ---
@app.route('/api/eventos')
def api_eventos():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Obtenemos los últimos 10 eventos
    cursor.execute("SELECT id, timestamp, nombre_detectado, tipo_evento, alerta_enviada FROM eventos ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    
    # Convertimos a lista de diccionarios para JSON
    eventos = []
    for row in rows:
        eventos.append({
            "id": row[0],
            "timestamp": row[1],
            "nombre": row[2],
            "tipo": row[3],
            "alerta": "SI" if row[4] == 1 else "NO"
        })
    return jsonify(eventos)

@app.route('/api/registrados')
def api_registrados():
    dataset_dir = "dataset"
    nombres = []
    
    # Verificamos que la carpeta exista
    if os.path.exists(dataset_dir):
        # Listamos solo los directorios (carpetas)
        for item in os.listdir(dataset_dir):
            if os.path.isdir(os.path.join(dataset_dir, item)):
                nombres.append(item)
    
    nombres.sort() # Los ordenamos alfabéticamente
    return jsonify(nombres)

if __name__ == "__main__":
    setup_database() # Crea la DB si no existe
    app.run(host="0.0.0.0", port=5000, debug=False)
