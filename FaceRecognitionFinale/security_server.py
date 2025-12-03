import os
import time
import cv2
import numpy as np
import smtplib
import threading
from flask import Flask, render_template, Response, jsonify
from email.mime.text import MIMEText
from picamera2 import Picamera2

# --- CONTROL DE SERVO (Tu código existente) ---
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory 

app = Flask(__name__)

# --- CONFIGURACIÓN ---
MODEL_DIR = "Object_Detection_Files" # Asegúrate de crear esta carpeta y poner los archivos
CLASS_FILE = os.path.join(MODEL_DIR, "coco.names")
CONFIG_PATH = os.path.join(MODEL_DIR, "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
WEIGHTS_PATH = os.path.join(MODEL_DIR, "frozen_inference_graph.pb")

# Objetos que nos interesa vigilar (Nombres exactos de coco.names)
TARGET_OBJECTS = ["keyboard", "mouse", "person", "tv", "desk"]

# Configuración de Correo (Misma que tu server.py)
EMAIL_SENDER = "latostada18@gmail.com"
EMAIL_PASSWORD = "prfe xusb oqgz layp"
EMAIL_RECEIVERS = ["caam314@gmail.com"]

# --- VARIABLES GLOBALES DE SEGURIDAD ---
SYSTEM_ARMED = False
REFERENCE_OBJECTS = {} # Aquí guardaremos la posición inicial: {'keyboard': (x,y), ...}
MISSING_OBJECT_TIMERS = {} # Para contar cuánto tiempo lleva perdido un objeto
MOVEMENT_THRESHOLD = 50 # Píxeles que se puede mover antes de alertar
DISAPPEAR_TIME_LIMIT = 5.0 # Segundos antes de enviar correo por robo
ALERTS_LOG = [] # Lista de alertas para la web

# --- INICIALIZACIÓN DE HARDWARE ---
# 1. Cámara
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# 2. Servos (Pan y Tilt)
factory = None
servo_pan = None
servo_tilt = None
current_pan = 0
current_tilt = 0

try:
    factory = PiGPIOFactory()
    # Servo Horizontal (GPIO 17)
    servo_pan = AngularServo(17, min_angle=-90, max_angle=90, 
                             min_pulse_width=0.0006, max_pulse_width=0.0024, pin_factory=factory)
    # Servo Vertical (GPIO 22)
    servo_tilt = AngularServo(22, min_angle=-90, max_angle=90, 
                              min_pulse_width=0.0006, max_pulse_width=0.0024, pin_factory=factory)
    servo_pan.angle = 0
    servo_tilt.angle = 0
    print("[INFO] Servos iniciados.")
except Exception as e:
    print(f"[ERROR] Servos: {e}")

# --- CARGAR MODELO DE IA ---
classNames = []
with open(CLASS_FILE, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

net = cv2.dnn_DetectionModel(WEIGHTS_PATH, CONFIG_PATH)
net.setInputSize(180, 180)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# --- FUNCIONES DE AYUDA ---

def send_security_email(subject, body):
    def _send():
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = EMAIL_SENDER
            msg['To'] = ", ".join(EMAIL_RECEIVERS)
            
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

def generate_frames():
    global ALERTS_LOG, REFERENCE_OBJECTS, MISSING_OBJECT_TIMERS
    
    while True:
        try:
            frame_xrgb = picam2.capture_array()
            img = cv2.cvtColor(frame_xrgb, cv2.COLOR_BGRA2BGR)
            
            # DETECCION DE OBJETOS
            classIds, confs, bbox = net.detect(img, confThreshold=0.20)
            
            current_detected_objects = {} # Objetos vistos en ESTE cuadro
            
            if len(classIds) != 0:
                for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    className = classNames[classId - 1]
                    print(f"[DEBUG] Veo: {className} ({int(confidence*100)}%)")
                    # Solo nos interesan los objetos de nuestra lista
                    if className in TARGET_OBJECTS:
                    # Guardamos su centro actual
                        center = get_center(box)
                        current_detected_objects[className] = center
                        
                        # Dibujar caja
                        color = (0, 255, 0) # Verde por defecto
                        label = f"{className.upper()} {int(confidence*100)}%"
                        
                        # LÓGICA DE SEGURIDAD (Si está ARMADO)
                        status_text = ""
                        if SYSTEM_ARMED:
                            # 1. Verificar si es un objeto vigilado que se movió
                            if className in REFERENCE_OBJECTS:
                                ref_center = REFERENCE_OBJECTS[className]
                                dist = np.linalg.norm(np.array(center) - np.array(ref_center))
                                
                                if dist > MOVEMENT_THRESHOLD:
                                    color = (0, 165, 255) # Naranja (Movido)
                                    status_text = " [MOVIDO!]"
                                    # Agregar alerta a la lista visual si no está reciente
                                    alert_msg = f"{className} se ha movido."
                                    if not ALERTS_LOG or ALERTS_LOG[-1] != alert_msg:
                                        ALERTS_LOG.append(alert_msg)
                                else:
                                    status_text = " [OK]"
                        
                        cv2.rectangle(img, box, color, 2)
                        cv2.putText(img, label + status_text, (box[0]+10, box[1]+30), 
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 2)

            # LÓGICA DE ROBO / DESAPARICIÓN
            if SYSTEM_ARMED:
                # Revisar cada objeto que DEBERÍA estar (Referencia)
                for obj_name in REFERENCE_OBJECTS:
                    # Si NO está en los detectados actualmente
                    if obj_name not in current_detected_objects:
                        # Empezar o continuar contador
                        if obj_name not in MISSING_OBJECT_TIMERS:
                            MISSING_OBJECT_TIMERS[obj_name] = time.time()
                        
                        elapsed = time.time() - MISSING_OBJECT_TIMERS[obj_name]
                        
                        # Alerta visual en pantalla (Texto Rojo)
                        cv2.putText(img, f"ALERTA: {obj_name.upper()} NO DETECTADO ({int(elapsed)}s)", 
                                    (10, 50 + (30 * list(REFERENCE_OBJECTS.keys()).index(obj_name))), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Si pasa el tiempo límite, enviar CORREO
                        if elapsed > DISAPPEAR_TIME_LIMIT:
                            # Verificar si ya enviamos correo recientemente para no hacer spam
                            # (Implementación simplificada: enviar y reiniciar timer muy largo)
                            if elapsed < DISAPPEAR_TIME_LIMIT + 2: # Solo envia una vez al cruzar el umbral
                                send_security_email(
                                    f"ALERTA ROBO: {obj_name} Desaparecido", 
                                    f"El objeto '{obj_name}' ha desaparecido de la mesa por mas de {DISAPPEAR_TIME_LIMIT} segundos."
                                )
                                ALERTS_LOG.append(f"ALERTA CRITICA: {obj_name} ROBADO - Email enviado.")
                    else:
                        # Si el objeto aparece, reiniciamos su timer de robo
                        if obj_name in MISSING_OBJECT_TIMERS:
                            del MISSING_OBJECT_TIMERS[obj_name]

            # Indicador de estado ARMADO/DESARMADO
            status_color = (0, 0, 255) if SYSTEM_ARMED else (255, 255, 0)
            status_txt = "SISTEMA ARMADO - VIGILANDO" if SYSTEM_ARMED else "SISTEMA DESARMADO - ESPERANDO"
            cv2.putText(img, status_txt, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

            # Enviar frame
            ret, buffer = cv2.imencode('.jpg', img)
            if not ret: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        except Exception as e:
            print(f"Error frame: {e}")

# --- RUTAS FLASK ---

@app.route('/')
def index():
    return render_template('security_dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/arm_system')
def arm_system():
    global SYSTEM_ARMED, REFERENCE_OBJECTS, MISSING_OBJECT_TIMERS, ALERTS_LOG
    
    # 1. Capturar una foto actual para ver qué hay
    frame_xrgb = picam2.capture_array()
    img = cv2.cvtColor(frame_xrgb, cv2.COLOR_BGRA2BGR)
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    
    REFERENCE_OBJECTS = {}
    MISSING_OBJECT_TIMERS = {}
    ALERTS_LOG = []
    
    detected_list = []
    
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in TARGET_OBJECTS:
                # Guardamos este objeto como referencia
                REFERENCE_OBJECTS[className] = get_center(box)
                detected_list.append(className)
    
    SYSTEM_ARMED = True
    ALERTS_LOG.append(f"Sistema ARMADO. Objetos registrados: {', '.join(detected_list)}")
    return jsonify({"status": "armed", "objects": detected_list})

@app.route('/disarm_system')
def disarm_system():
    global SYSTEM_ARMED
    SYSTEM_ARMED = False
    return jsonify({"status": "disarmed"})

@app.route('/get_alerts')
def get_alerts():
    # Devuelve los ultimos 5 eventos
    return jsonify(ALERTS_LOG[-10:])

# --- RUTAS DE SERVO (Copiadas para tener control manual) ---
@app.route('/move_servo/<direction>')
def move_servo(direction):
    global current_pan
    step = 10
    if direction == 'left': new_angle = current_pan + step
    elif direction == 'right': new_angle = current_pan - step
    elif direction == 'center': new_angle = 0
    else: return jsonify({"status": "error"})
    
    if new_angle > 90: new_angle = 90
    if new_angle < -90: new_angle = -90
    
    if servo_pan:
        servo_pan.angle = new_angle
        current_pan = new_angle
    return jsonify({"status": "ok"})

@app.route('/move_servo_v/<direction>')
def move_servo_v(direction):
    global current_tilt
    step = 10
    if direction == 'up': new_angle = current_tilt + step
    elif direction == 'down': new_angle = current_tilt - step
    elif direction == 'center': new_angle = 0
    else: return jsonify({"status": "error"})
    
    if new_angle > 90: new_angle = 90
    if new_angle < -90: new_angle = -90
    
    if servo_tilt:
        servo_tilt.angle = new_angle
        current_tilt = new_angle
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=False) 
    # NOTA: Usamos puerto 5001 para que no choque con tu otro server