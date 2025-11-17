import subprocess
import threading
import os
import sys

import face_recognition
import pickle
import numpy as np
import datetime
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

from picamera2 import Picamera2
import time

from flask import Flask, render_template, Response, request, jsonify
import cv2, pickle, sqlite3, os, datetime

app = Flask(__name__)

ENCODINGS_FILE = "encodings.pickle"
DB_NAME = "log_eventos.db"

# Lista de correos
EMAIL_RECEIVERS = []

# ---------------- Funciones auxiliares ----------------
def cargar_nombres():
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.loads(f.read())
            return list(set(data["names"]))
    return []

def cargar_eventos():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, nombre_detectado FROM eventos ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()
    return rows

# ---------------- Video ----------------
# Configuración de la Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()
# ... (justo después de picam2.start()) ...

# Variable para controlar la detección
DETECTION_ENABLED = True

# Cargar los encodings (como en facial_recognition.py)
print("[INFO] Cargando encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# --- CONSTANTES DE CODIGO_NUEVO.PY ---

# ¡¡ADVERTENCIA!! Rellena tus credenciales de correo
# Necesitas una "Contraseña de aplicación" de Google, no tu contraseña normal
EMAIL_SENDER = "latostada18@gmail.com"
EMAIL_PASSWORD = "prfe xusb oqgz layp"
EMAIL_RECEIVERS = ["caam314@gmail.com"] # Lista de correos

KNOWN_FACES_DIR = "capturas_conocidos"
UNKNOWN_FACES_DIR = "capturas_desconocidos"
DETECTION_COOLDOWN = 10 # segundos
EMAIL_COOLDOWN = 60 * 5 # 5 minutos

CURRENT_MODE = "inactive"
CURRENT_STATUS_MSG = ""
CAPTURE_NAME = ""
CAPTURE_COUNT = 0
MAX_CAPTURES = 10
LAST_CAPTURE_TIME = 0

TRAINING_STATUS = ""

# Asegúrate de que estas carpetas existan
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)
if not os.path.exists(UNKNOWN_FACES_DIR):
    os.makedirs(UNKNOWN_FACES_DIR)

# Lista de personas que NO deben generar alerta
authorized_names = ["carlos", "valeria", "marcos","anali", "ian"] # Reemplaza con tus nombres autorizados

# Diccionarios para manejar los cooldowns
last_detection_time = {}
last_email_sent_time = {}

# Variables para el proceso de reconocimiento
face_locations = []
face_names = []
cv_scaler = 4 # Escala para procesar más rápido

time.sleep(1.0) # Darle un segundo a la cámara para que arranque

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
        
        # Adjuntar imagen
        attachment = open(image_path, "rb")
        part = MIMEBase('application', 'octet-stream')
        part.set_payload((attachment).read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', "attachment; filename= %s" % os.path.basename(image_path))
        msg.attach(part)
        
        # Enviar correo
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVERS, text)
        server.quit()
        print(f"[INFO] Email de alerta enviado exitosamente para {name}.")
    except Exception as e:
        print(f"[ERROR] No se pudo enviar el email: {e}")

def save_capture(frame_bgr, name):
    # Asegúrate de que el frame esté en formato BGR antes de llamar a esto
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    if name == "Unknown":
        folder = UNKNOWN_FACES_DIR
        filename = f"unknown_{timestamp}.jpg"
    else:
        folder = KNOWN_FACES_DIR
        filename = f"{name.lower()}_{timestamp}.jpg"
        
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame_bgr)
    print(f"[INFO] Captura guardada en: {filepath}")
    return filepath

def generar_video():
    # Declaramos todas las variables globales que usaremos
    global face_locations, face_names, last_detection_time, last_email_sent_time
    global CURRENT_MODE, CURRENT_STATUS_MSG, CAPTURE_NAME, CAPTURE_COUNT, LAST_CAPTURE_TIME
    global known_face_encodings, known_face_names

    while True:
        # 1. Capturar y convertir el frame (esto pasa en todos los modos)
        frame_xrgb = picam2.capture_array()
        # Convertimos a BGR, que es el formato que usa OpenCV para dibujar
        frame_bgr = cv2.cvtColor(frame_xrgb, cv2.COLOR_RGBA2BGR)
        # Hacemos una copia para dibujar encima. El frame_bgr original se usa para guardar
        display_frame = frame_bgr.copy()
        current_time = time.time()
        
        # =======================================================
        # === MODO 1: DETECCIÓN (detection)
        # =======================================================
        if CURRENT_MODE == "detection":
            # --- Lógica de Reconocimiento (de codigo_nuevo.py) ---
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
                
                # --- Lógica de Cooldown y Alerta por Email ---
                if current_time - last_detection_time.get(name, 0) > DETECTION_COOLDOWN:
                    last_detection_time[name] = current_time
                    # Guardamos el frame original BGR de alta calidad
                    image_path = save_capture(frame_bgr, name)
                    
                    if (name == "Unknown" or name not in authorized_names) and (current_time - last_email_sent_time.get(name, 0) > EMAIL_COOLDOWN):
                        last_email_sent_time[name] = current_time
                        print(f"[INFO] Enviando alerta por email para: {name}")
                        email_thread = threading.Thread(target=send_email_alert, args=(name, image_path))
                        email_thread.start()

            # --- Lógica de Dibujo (para el modo Detección) ---
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= cv_scaler
                right *= cv_scaler
                bottom *= cv_scaler
                left *= cv_scaler
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), (244, 42, 3), 3)
                cv2.rectangle(display_frame, (left -3, top - 35), (right+3, top), (244, 42, 3), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(display_frame, name, (left + 6, top - 6), font, 1.0, (255, 255, 255), 1)
                
                if name in authorized_names:
                    cv2.putText(display_frame, "Autorizado", (left + 6, bottom + 23), font, 0.6, (0, 255, 0), 1)

        # =======================================================
        # === MODO 2: CAPTURA (capture)
        # =======================================================
        elif CURRENT_MODE == "capture":
            # Tomamos 1 foto por segundo
            if (current_time - LAST_CAPTURE_TIME > 1): 
                LAST_CAPTURE_TIME = current_time
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{CAPTURE_NAME}_{timestamp}.jpg"
                filepath = os.path.join("dataset", CAPTURE_NAME, filename)
                cv2.imwrite(filepath, frame_bgr) # Guardamos el frame BGR
                
                CAPTURE_COUNT += 1
                CURRENT_STATUS_MSG = f"Capturando foto {CAPTURE_COUNT}/{MAX_CAPTURES}..."
                print(f"Foto {CAPTURE_COUNT}/{MAX_CAPTURES} guardada para {CAPTURE_NAME}")

            # Dibujar indicador de captura en el video
            cv2.putText(display_frame, f"CAPTURANDO: {CAPTURE_COUNT}/{MAX_CAPTURES}", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # ¿Terminamos de capturar?
            if CAPTURE_COUNT >= MAX_CAPTURES:
                CURRENT_MODE = "training" # Cambiamos a modo "Entrenamiento"
                CURRENT_STATUS_MSG = "Captura finalizada. Re-entrenando..."
                print("Iniciando re-entrenamiento en segundo plano...")
                
                # --- Función de entrenamiento (se ejecuta en un hilo) ---
                def train_model():
                    global CURRENT_MODE, CURRENT_STATUS_MSG, known_face_encodings, known_face_names
                    try:
                        subprocess.run(["python", "model_training.py"], check=True)
                        print("Re-entrenamiento finalizado.")
                        
                        # ¡IMPORTANTE! Recargar los encodings
                        with open("encodings.pickle", "rb") as f:
                            data = pickle.loads(f.read())
                        known_face_encodings = data["encodings"]
                        known_face_names = data["names"]
                        
                        CURRENT_STATUS_MSG = "¡Entrenamiento completado!"
                        time.sleep(5) # Muestra el mensaje por 5 segundos
                        CURRENT_STATUS_MSG = ""
                        CURRENT_MODE = "detection" # ¡Volvemos al modo detección!
                    except Exception as e:
                        print(f"Error en re-entrenamiento: {e}")
                        CURRENT_STATUS_MSG = "Error de entrenamiento. Volviendo a detección."
                        time.sleep(5)
                        CURRENT_STATUS_MSG = ""
                        CURRENT_MODE = "detection"
                # --- Fin de la función del hilo ---
                
                # Iniciar el hilo
                train_thread = threading.Thread(target=train_model)
                train_thread.start()
        
        # =======================================================
        # === MODO 3: ENTRENAMIENTO (training)
        # =======================================================
        elif CURRENT_MODE == "training":
            # Solo mostramos un texto mientras el hilo trabaja
            cv2.putText(display_frame, "ENTRENANDO...", 
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
            
        # =======================================================
        # === MODO 4: INACTIVO (inactive)
        # =======================================================
        # (Si el modo es "inactive", no entra a ningún 'if' y solo
        # muestra el video limpio, sin dibujos)

        # 5. Codificar y transmitir (siempre)
        # 'display_frame' tendrá los dibujos (si los hay) o será el frame limpio
        ret, buffer = cv2.imencode('.jpg', display_frame)
        if not ret:
            continue
            
        frame_bytes = buffer.tobytes()
        
        # Enviar el frame al navegador
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


# ---------------- Rutas Flask ----------------



@app.route('/')
def dashboard():
    nombres = cargar_nombres()
    eventos = cargar_eventos()
    ultimo_evento = eventos[0][1] if eventos else "Ninguno"
    return render_template("dashboard.html", nombres=nombres, eventos=eventos,
                           ultimo_evento=ultimo_evento, correos=EMAIL_RECEIVERS)

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
        face_locations = [] # Limpia los recuadros al detener
        face_names = []
    print(f"Cambiando modo a: {CURRENT_MODE}")
    return jsonify({"new_mode": CURRENT_MODE})

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global CURRENT_MODE, CAPTURE_NAME, CAPTURE_COUNT, CURRENT_STATUS_MSG
    
    name = request.form.get("name")
    if not name:
        return jsonify({"status": "Error"}), 400

    # 1. Preparamos las variables
    CAPTURE_NAME = name
    CAPTURE_COUNT = 0
    CURRENT_MODE = "capture" # ¡Activamos el modo captura!
    CURRENT_STATUS_MSG = f"Iniciando captura para {CAPTURE_NAME}..."
    
    # 2. Creamos la carpeta
    dataset_folder = os.path.join("dataset", name)
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        
    print(f"MODO CAPTURA: Iniciado para '{name}'.")
    return jsonify({"status": "ok"})

@app.route('/get_status')
def get_status():
    global CURRENT_MODE, CURRENT_STATUS_MSG
    return jsonify({"mode": CURRENT_MODE, "message": CURRENT_STATUS_MSG})



@app.route('/borrar_correo', methods=['POST'])
def borrar_correo():
    correo = request.form.get("correo")
    if correo in EMAIL_RECEIVERS:
        EMAIL_RECEIVERS.remove(correo)
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
