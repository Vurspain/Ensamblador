import cv2
import face_recognition
import pickle
import numpy as np
import datetime
import os
import sys
import threading
import smtplib
import sqlite3
import time
from flask import Flask, render_template, Response, request, jsonify
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from picamera2 import Picamera2

# --- CONTROL DE HARDWARE ---
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory 

app = Flask(__name__)

# ==========================================
# === CONFIGURACIÓN GENERAL ===
# ==========================================

# 1. Rutas y Archivos
ENCODINGS_FILE = "encodings.pickle"
DB_NAME = "log_eventos.db"
KNOWN_FACES_DIR = "capturas_conocidos"
UNKNOWN_FACES_DIR = "capturas_desconocidos"
PATROL_DIR = "capturas_patrulla"

# Directorio para IA de Objetos
OBJ_MODEL_DIR = "Object_Detection_Files" # ¡Asegúrate que exista!
OBJ_CLASS_FILE = os.path.join(OBJ_MODEL_DIR, "coco.names")
OBJ_CONFIG = os.path.join(OBJ_MODEL_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
OBJ_WEIGHTS = os.path.join(OBJ_MODEL_DIR, "frozen_inference_graph.pb")

# 2. Configuración de Correo
EMAIL_SENDER = "latostada18@gmail.com"
EMAIL_PASSWORD = "prfe xusb oqgz layp"
EMAIL_RECEIVERS = ["caam314@gmail.com"] 

# 3. Variables de Control Global
SYSTEM_MODE = "INACTIVE" # Modos: "INACTIVE", "FACE_REC", "FACE_CAPTURE", "FACE_PATROL", "OBJECT_SEC"
CURRENT_STATUS_MSG = "Sistema en espera..."

# ==========================================
# === INICIALIZACIÓN DE HARDWARE ===
# ==========================================

# 1. Asegurar Carpetas
for folder in [KNOWN_FACES_DIR, UNKNOWN_FACES_DIR, PATROL_DIR]:
    if not os.path.exists(folder): os.makedirs(folder)

# 2. Cámara
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# 3. Servos (Pan/Tilt)
factory = None
servo_pan = None
servo_tilt = None
current_pan = 0
current_tilt = 0

try:
    factory = PiGPIOFactory()
    servo_pan = AngularServo(17, min_angle=-90, max_angle=90, min_pulse_width=0.0006, max_pulse_width=0.0024, pin_factory=factory)
    servo_tilt = AngularServo(22, min_angle=-90, max_angle=90, min_pulse_width=0.0006, max_pulse_width=0.0024, pin_factory=factory)
    servo_pan.angle = 0
    servo_tilt.angle = 0
    print("[INFO] Servos iniciados.")
except Exception as e:
    print(f"[ERROR] Servos: {e}")

# ==========================================
# === CARGA DE MODELOS DE INTELIGENCIA ===
# ==========================================

# 1. Cargar Reconocimiento Facial
print("[INFO] Cargando Base de Datos Facial...")
known_face_encodings = []
known_face_names = []
if os.path.exists(ENCODINGS_FILE):
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.loads(f.read())
    known_face_encodings = data["encodings"]
    known_face_names = data["names"]

# 2. Cargar Detección de Objetos (MobileNet)
print("[INFO] Cargando Detector de Objetos...")
obj_classNames = []
if os.path.exists(OBJ_CLASS_FILE):
    with open(OBJ_CLASS_FILE, "rt") as f:
        obj_classNames = f.read().rstrip("\n").split("\n")

obj_net = cv2.dnn_DetectionModel(OBJ_WEIGHTS, OBJ_CONFIG)
obj_net.setInputSize(320, 320)
obj_net.setInputScale(1.0 / 127.5)
obj_net.setInputMean((127.5, 127.5, 127.5))
obj_net.setInputSwapRB(True)

# Objetos a vigilar
TARGET_OBJECTS = ["keyboard", "mouse", "tvmonitor", "laptop", "cell phone", "person"]

# ==========================================
# === VARIABLES ESPECÍFICAS DE MODOS ===
# ==========================================

# Variables para Rostros
authorized_names = ["carlos", "valeria", "marcos", "anali", "ian"]
last_detection_time = {}
last_email_sent_time = {}
face_locations = []
face_names = []
DETECTION_COOLDOWN = 10 
EMAIL_COOLDOWN = 300

# Variables para Captura (Entrenamiento)
CAPTURE_NAME = ""
CAPTURE_COUNT = 0
MAX_CAPTURES = 10
LAST_CAPTURE_TIME = 0

# Variables para Patrulla
LAST_PATROL_TIME = 0
PATROL_INTERVAL = 5
PATROL_POSITION = 0

# Variables para Objetos (Seguridad)
OBJ_ARMED = False
REFERENCE_OBJECTS = {} 
MISSING_OBJECT_TIMERS = {}
MOVEMENT_THRESHOLD = 50 
DISAPPEAR_TIME_LIMIT = 5.0 
ALERTS_LOG = [] 

# ==========================================
# === FUNCIONES AUXILIARES ===
# ==========================================

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
        sql = "INSERT INTO eventos (timestamp, nombre_detectado, tipo_evento, nombre_clip, alerta_enviada) VALUES (?, ?, ?, ?, ?)"
        cursor.execute(sql, (timestamp, name, event_type, filename, alert_sent))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[ERROR DB] {e}")

def send_email_alert(subject, body, image_path=None):
    def _send():
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_SENDER
            msg['To'] = ", ".join(EMAIL_RECEIVERS)
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            if image_path:
                attachment = open(image_path, "rb")
                part = MIMEBase('application', 'octet-stream')
                part.set_payload((attachment).read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition', "attachment; filename= %s" % os.path.basename(image_path))
                msg.attach(part)
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.sendmail(EMAIL_SENDER, EMAIL_RECEIVERS, msg.as_string())
            server.quit()
            print(f"[EMAIL] Enviado: {subject}")
        except Exception as e:
            print(f"[ERROR EMAIL] {e}")
    threading.Thread(target=_send).start()

def get_center(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)

def mover_servo(pan, tilt):
    global current_pan, current_tilt
    # Limitar ángulos
    if pan > 90: pan = 90
    if pan < -90: pan = -90
    if tilt > 90: tilt = 90
    if tilt < -90: tilt = -90
    
    if servo_pan: servo_pan.angle = pan
    if servo_tilt: servo_tilt.angle = tilt
    
    current_pan = pan
    current_tilt = tilt

# ==========================================
# === BUCLE PRINCIPAL DE VIDEO ===
# ==========================================

def generate_frames():
    global SYSTEM_MODE, CURRENT_STATUS_MSG, ALERTS_LOG
    global face_locations, face_names, last_detection_time, last_email_sent_time
    global CAPTURE_NAME, CAPTURE_COUNT, LAST_CAPTURE_TIME, known_face_encodings, known_face_names
    global LAST_PATROL_TIME, PATROL_POSITION
    global OBJ_ARMED, REFERENCE_OBJECTS, MISSING_OBJECT_TIMERS

    while True:
        try:
            frame_xrgb = picam2.capture_array()
            # Usar BGRA2BGR para colores correctos
            frame_bgr = cv2.cvtColor(frame_xrgb, cv2.COLOR_BGRA2BGR)
            display_frame = frame_bgr.copy()
            current_time = time.time()

            # ---------------------------------------------------------
            # MODO 1: RECONOCIMIENTO FACIAL (Y PATRULLA)
            # ---------------------------------------------------------
            if SYSTEM_MODE in ["FACE_REC", "FACE_PATROL"]:
                
                # Lógica Patrulla (Solo si está activo)
                if SYSTEM_MODE == "FACE_PATROL":
                    if current_time - LAST_PATROL_TIME > PATROL_INTERVAL:
                        LAST_PATROL_TIME = current_time
                        if PATROL_POSITION == 0:
                            mover_servo(45, 0)
                            PATROL_POSITION = 1
                        else:
                            mover_servo(-45, 0)
                            PATROL_POSITION = 0
                        # Guardar foto patrulla
                        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        cv2.imwrite(os.path.join(PATROL_DIR, f"patrulla_{ts}.jpg"), frame_bgr)
                    cv2.putText(display_frame, "MODO PATRULLA", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Lógica Detección Facial
                small_frame = cv2.resize(frame_bgr, (0, 0), fx=0.25, fy=0.25)
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
                face_names = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]
                    face_names.append(name)

                    # Eventos y Alertas
                    if current_time - last_detection_time.get(name, 0) > DETECTION_COOLDOWN:
                        last_detection_time[name] = current_time
                        ts_db = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Guardar foto
                        ts_file = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        folder = KNOWN_FACES_DIR if name != "Unknown" else UNKNOWN_FACES_DIR
                        fname = f"{name}_{ts_file}.jpg"
                        fpath = os.path.join(folder, fname)
                        cv2.imwrite(fpath, frame_bgr)

                        alert_sent = 0
                        if (name == "Unknown" or name not in authorized_names) and (current_time - last_email_sent_time.get(name, 0) > EMAIL_COOLDOWN):
                            last_email_sent_time[name] = current_time
                            alert_sent = 1
                            send_email_alert(f"ALERTA: {name} Detectado", f"Se detectó a {name}.", fpath)
                        
                        threading.Thread(target=log_event_to_db, args=(ts_db, name, "Detección", fname, alert_sent)).start()

                # Dibujar Cajas
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    top *= 4; right *= 4; bottom *= 4; left *= 4
                    color = (0, 255, 0) if name in authorized_names else (0, 0, 255)
                    cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                    cv2.putText(display_frame, name, (left, top - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 1)

            # ---------------------------------------------------------
            # MODO 2: CAPTURA DE ROSTROS
            # ---------------------------------------------------------
            elif SYSTEM_MODE == "FACE_CAPTURE":
                if current_time - LAST_CAPTURE_TIME > 1:
                    LAST_CAPTURE_TIME = current_time
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"{CAPTURE_NAME}_{ts}.jpg"
                    fpath = os.path.join("dataset", CAPTURE_NAME, fname)
                    cv2.imwrite(fpath, frame_bgr)
                    CAPTURE_COUNT += 1
                    CURRENT_STATUS_MSG = f"Capturando {CAPTURE_COUNT}/{MAX_CAPTURES}..."
                
                cv2.putText(display_frame, f"CAPTURANDO: {CAPTURE_COUNT}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

                if CAPTURE_COUNT >= MAX_CAPTURES:
                    CURRENT_STATUS_MSG = "Entrenando..."
                    # Entrenar en hilo para no bloquear video
                    def train():
                        global SYSTEM_MODE, known_face_encodings, known_face_names, CURRENT_STATUS_MSG
                        import subprocess
                        subprocess.run(["python", "model_training.py"])
                        with open(ENCODINGS_FILE, "rb") as f:
                            data = pickle.loads(f.read())
                        known_face_encodings = data["encodings"]
                        known_face_names = data["names"]
                        CURRENT_STATUS_MSG = "¡Listo!"
                        time.sleep(2)
                        SYSTEM_MODE = "FACE_REC"
                    threading.Thread(target=train).start()

            # ---------------------------------------------------------
            # MODO 3: SEGURIDAD DE OBJETOS
            # ---------------------------------------------------------
            elif SYSTEM_MODE == "OBJECT_SEC":
                # Usar MobileNet
                classIds, confs, bbox = obj_net.detect(display_frame, confThreshold=0.30)
                current_objects = {}

                if len(classIds) != 0:
                    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                        if classId - 1 < len(obj_classNames):
                            className = obj_classNames[classId - 1]
                            
                            if className in TARGET_OBJECTS:
                                center = get_center(box)
                                current_objects[className] = center
                                
                                color = (255, 0, 0) # Azul
                                status_text = ""

                                # Si está ARMADO, verificar movimiento
                                if OBJ_ARMED:
                                    if className in REFERENCE_OBJECTS:
                                        ref = REFERENCE_OBJECTS[className]
                                        dist = np.linalg.norm(np.array(center) - np.array(ref))
                                        if dist > MOVEMENT_THRESHOLD:
                                            color = (0, 165, 255) # Naranja
                                            status_text = " [MOVIDO]"
                                            msg = f"ALERTA: {className} movido."
                                            if not ALERTS_LOG or ALERTS_LOG[-1] != msg: ALERTS_LOG.append(msg)
                                
                                cv2.rectangle(display_frame, box, color, 2)
                                cv2.putText(display_frame, f"{className} {status_text}", (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Si está ARMADO, verificar robo
                if OBJ_ARMED:
                    for obj in REFERENCE_OBJECTS:
                        if obj not in current_objects:
                            if obj not in MISSING_OBJECT_TIMERS:
                                MISSING_OBJECT_TIMERS[obj] = time.time()
                            
                            elapsed = time.time() - MISSING_OBJECT_TIMERS[obj]
                            cv2.putText(display_frame, f"FALTA: {obj} ({int(elapsed)}s)", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

                            if elapsed > DISAPPEAR_TIME_LIMIT and elapsed < DISAPPEAR_TIME_LIMIT + 1:
                                msg = f"ROBO DETECTADO: {obj}"
                                ALERTS_LOG.append(msg)
                                send_email_alert("ALERTA DE ROBO", f"El objeto {obj} ha desaparecido.")
                        else:
                            if obj in MISSING_OBJECT_TIMERS: del MISSING_OBJECT_TIMERS[obj]
                
                status_txt = "OBJETOS: ARMADO" if OBJ_ARMED else "OBJETOS: DESARMADO"
                cv2.putText(display_frame, status_txt, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            # ---------------------------------------------------------
            # MODO 4: INACTIVO
            # ---------------------------------------------------------
            else:
                cv2.putText(display_frame, "SISTEMA INACTIVO", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100,100,100), 2)

            # ENVIAR FRAME
            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Error loop: {e}")

# ==========================================
# === RUTAS DE LA API (FLASK) ===
# ==========================================

@app.route('/')
def index():
    return render_template('master_dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# --- CONTROL PRINCIPAL DEL SISTEMA ---
@app.route('/set_system/<mode>')
def set_system(mode):
    global SYSTEM_MODE, CURRENT_STATUS_MSG
    # Reiniciar estados al cambiar
    if mode == "FACE_REC":
        SYSTEM_MODE = "FACE_REC"
        CURRENT_STATUS_MSG = "Modo: Reconocimiento Facial"
    elif mode == "OBJECT_SEC":
        SYSTEM_MODE = "OBJECT_SEC"
        CURRENT_STATUS_MSG = "Modo: Seguridad de Objetos"
    elif mode == "INACTIVE":
        SYSTEM_MODE = "INACTIVE"
        CURRENT_STATUS_MSG = "Sistema Detenido"
    return jsonify({"status": "ok", "mode": SYSTEM_MODE})

# --- RUTAS DE ROSTROS ---
@app.route('/toggle_patrol')
def toggle_patrol():
    global SYSTEM_MODE
    if SYSTEM_MODE == "FACE_REC": SYSTEM_MODE = "FACE_PATROL"
    elif SYSTEM_MODE == "FACE_PATROL": SYSTEM_MODE = "FACE_REC"
    return jsonify({"status": "ok", "mode": SYSTEM_MODE})

@app.route('/start_capture', methods=['POST'])
def start_capture():
    global SYSTEM_MODE, CAPTURE_NAME, CAPTURE_COUNT
    name = request.form.get("name")
    if name:
        CAPTURE_NAME = name
        folder = os.path.join("dataset", name)
        if not os.path.exists(folder): os.makedirs(folder)
        CAPTURE_COUNT = 0
        SYSTEM_MODE = "FACE_CAPTURE"
        return jsonify({"status": "ok"})
    return jsonify({"status": "error"}), 400

@app.route('/api/eventos')
def api_eventos():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT timestamp, nombre_detectado, tipo_evento, alerta_enviada FROM eventos ORDER BY id DESC LIMIT 10")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{"time":r[0], "name":r[1], "type":r[2], "alert":r[3]} for r in rows])

@app.route('/api/registrados')
def api_registrados():
    nombres = []
    if os.path.exists("dataset"):
        nombres = [d for d in os.listdir("dataset") if os.path.isdir(os.path.join("dataset", d))]
    return jsonify(nombres)

# --- RUTAS DE OBJETOS ---
@app.route('/arm_objects')
def arm_objects():
    global OBJ_ARMED, REFERENCE_OBJECTS, ALERTS_LOG
    # Tomar foto de referencia
    frame_xrgb = picam2.capture_array()
    img = cv2.cvtColor(frame_xrgb, cv2.COLOR_BGRA2BGR)
    classIds, confs, bbox = obj_net.detect(img, confThreshold=0.30)
    
    REFERENCE_OBJECTS = {}
    detected = []
    if len(classIds) != 0:
        for classId, _, box in zip(classIds.flatten(), confs.flatten(), bbox):
            if classId - 1 < len(obj_classNames):
                name = obj_classNames[classId - 1]
                if name in TARGET_OBJECTS:
                    REFERENCE_OBJECTS[name] = get_center(box)
                    detected.append(name)
    
    OBJ_ARMED = True
    msg = f"Sistema ARMADO. Objetos: {', '.join(detected)}"
    if len(ALERTS_LOG) > 0 and ALERTS_LOG[-1] != msg: ALERTS_LOG.append(msg)
    elif len(ALERTS_LOG) == 0: ALERTS_LOG.append(msg)
    
    return jsonify({"status": "armed", "objects": detected})

@app.route('/disarm_objects')
def disarm_objects():
    global OBJ_ARMED
    OBJ_ARMED = False
    return jsonify({"status": "disarmed"})

@app.route('/get_object_alerts')
def get_object_alerts():
    return jsonify(ALERTS_LOG[-10:])

# --- CONTROL MANUAL DE SERVOS ---
@app.route('/move_servo/<axis>/<direction>')
def move_servo_route(axis, direction):
    global current_pan, current_tilt
    step = 10
    
    if axis == "pan":
        if direction == "left": current_pan += step
        elif direction == "right": current_pan -= step
        elif direction == "center": current_pan = 0
    elif axis == "tilt":
        if direction == "up": current_tilt += step
        elif direction == "down": current_tilt -= step
        elif direction == "center": current_tilt = 0
    
    mover_servo(current_pan, current_tilt)
    return jsonify({"status": "ok"})

# --- MAIN ---
if __name__ == "__main__":
    setup_database()
    app.run(host="0.0.0.0", port=5000, debug=False)