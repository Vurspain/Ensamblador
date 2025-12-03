import cv2
import numpy as np
import time
import os
import threading
import smtplib
from flask import Flask, render_template, Response, jsonify
from email.mime.text import MIMEText
from picamera2 import Picamera2

# --- CONTROL DE HARDWARE (Tu configuración probada) ---
from gpiozero import AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory

app = Flask(__name__)

# --- CONFIGURACIÓN YOLO ---
MODEL_DIR = "YOLO_Files"  # Carpeta donde guardaste los archivos
LABELS_FILE = os.path.join(MODEL_DIR, "coco.names")
CONFIG_FILE = os.path.join(MODEL_DIR, "yolov4-tiny.cfg")
WEIGHTS_FILE = os.path.join(MODEL_DIR, "yolov4-tiny.weights")

# Objetos a vigilar (Nombres exactos de coco.names)
TARGET_OBJECTS = ["keyboard", "mouse", "tvmonitor", "laptop", "cell phone", "person"]

# Configuración de Correo
EMAIL_SENDER = "latostada18@gmail.com"
EMAIL_PASSWORD = "prfe xusb oqgz layp"
EMAIL_RECEIVERS = ["caam314@gmail.com"]

# --- VARIABLES DE ESTADO ---
SYSTEM_ARMED = False
REFERENCE_POSITIONS = {}    # Dónde estaban las cosas al armar
MISSING_TIMERS = {}         # Contador para objetos robados
MOVEMENT_THRESHOLD = 40     # Píxeles de tolerancia para movimiento
DISAPPEAR_LIMIT = 5.0       # Segundos antes de alertar robo
ALERTS_LOG = []

# --- INICIALIZACIÓN HARDWARE ---
# 1. Cámara
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

# 2. Servos (Pan/Tilt) con pigpio
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
    
    # Posición inicial
    servo_pan.angle = 0
    servo_tilt.angle = 0
    print("[INFO] Servos Pan/Tilt iniciados con Hardware PWM.")
except Exception as e:
    print(f"[ERROR] No se pudieron iniciar los servos: {e}")

# --- CARGAR YOLO ---
print("[INFO] Cargando YOLO v4 Tiny...")
net = cv2.dnn.readNetFromDarknet(CONFIG_FILE, WEIGHTS_FILE)
# Usar CPU (La Pi 4 no tiene GPU compatible con CUDA)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Cargar etiquetas
labels = open(LABELS_FILE).read().strip().split("\n")

# Obtener las capas de salida
ln = net.getLayerNames()
try:
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
except:
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# --- FUNCIONES ---

def send_alert_email(subject, body):
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

def get_center(x, y, w, h):
    return (x + w // 2, y + h // 2)

def detect_objects(img):
    """Función central de detección YOLO"""
    (H, W) = img.shape[:2]
    
    # Preprocesamiento para YOLO (Escala 1/255, Tamaño 416x416, SwapRB=True)
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (608, 608), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    # Procesar salidas
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # Umbral de confianza (0.3 = 30%)
            if confidence > 0.3:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Non-Maxima Suppression (Eliminar cajas duplicadas)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.3)
    
    results = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            results.append({
                "label": labels[classIDs[i]],
                "confidence": confidences[i],
                "box": boxes[i]
            })
    return results

def generate_frames():
    global ALERTS_LOG, REFERENCE_POSITIONS, MISSING_TIMERS

    while True:
        try:
            # 1. Capturar Frame
            frame_xrgb = picam2.capture_array()
            img = cv2.cvtColor(frame_xrgb, cv2.COLOR_BGRA2BGR)
            
            # 2. Detectar con YOLO
            detections = detect_objects(img)
            
            current_objects = {}

            # 3. Procesar detecciones
            for obj in detections:
                label = obj["label"]
                (x, y, w, h) = obj["box"]
                
                # Solo procesar si está en nuestra lista de interés
                if label in TARGET_OBJECTS:
                    center = get_center(x, y, w, h)
                    current_objects[label] = center
                    
                    color = (0, 255, 0) # Verde
                    status = ""

                    # Lógica de Seguridad (Si está ARMADO)
                    if SYSTEM_ARMED:
                        if label in REFERENCE_POSITIONS:
                            ref_center = REFERENCE_POSITIONS[label]
                            dist = np.linalg.norm(np.array(center) - np.array(ref_center))
                            
                            if dist > MOVEMENT_THRESHOLD:
                                color = (0, 165, 255) # Naranja (Movido)
                                status = " [MOVIDO]"
                                msg = f"ALERTA: {label} se ha movido."
                                if not ALERTS_LOG or ALERTS_LOG[-1] != msg:
                                    ALERTS_LOG.append(msg)
                    
                    # Dibujar
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    text = f"{label}: {int(obj['confidence']*100)}%{status}"
                    cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # 4. Verificar Objetos Faltantes (ROBO)
            if SYSTEM_ARMED:
                for target_name in REFERENCE_POSITIONS:
                    if target_name not in current_objects:
                        # Objeto perdido
                        if target_name not in MISSING_TIMERS:
                            MISSING_TIMERS[target_name] = time.time()
                        
                        elapsed = time.time() - MISSING_TIMERS[target_name]
                        
                        # Alerta visual ROJA
                        cv2.putText(img, f"FALTA: {target_name.upper()} ({int(elapsed)}s)", 
                                    (10, 50 + (30 * list(REFERENCE_POSITIONS.keys()).index(target_name))),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        if elapsed > DISAPPEAR_LIMIT:
                            if elapsed < DISAPPEAR_LIMIT + 1.5: # Enviar solo una vez
                                send_alert_email(f"ALERTA ROBO: {target_name}", f"El objeto {target_name} ha desaparecido.")
                                ALERTS_LOG.append(f"CRITICO: {target_name} DESAPARECIDO - Email enviado.")
                    else:
                        # Objeto encontrado, resetear timer
                        if target_name in MISSING_TIMERS:
                            del MISSING_TIMERS[target_name]

            # Estado del sistema
            txt_state = "SISTEMA ARMADO (YOLO)" if SYSTEM_ARMED else "SISTEMA DESARMADO"
            color_state = (0, 0, 255) if SYSTEM_ARMED else (255, 255, 0)
            cv2.putText(img, txt_state, (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_state, 2)

            ret, buffer = cv2.imencode('.jpg', img)
            if not ret: continue
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

        except Exception as e:
            print(f"Error loop: {e}")

# --- RUTAS FLASK (API) ---

@app.route('/')
def index():
    return render_template('security_dashboard.html') # Usamos el mismo HTML de antes

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/arm_system')
def arm_system():
    global SYSTEM_ARMED, REFERENCE_POSITIONS, MISSING_TIMERS, ALERTS_LOG
    
    # Captura única para calibrar
    frame_xrgb = picam2.capture_array()
    img = cv2.cvtColor(frame_xrgb, cv2.COLOR_BGRA2BGR)
    
    # Detección inicial
    detections = detect_objects(img)
    
    REFERENCE_POSITIONS = {}
    MISSING_TIMERS = {}
    ALERTS_LOG = []
    detected_names = []
    
    for obj in detections:
        label = obj["label"]
        if label in TARGET_OBJECTS:
            (x, y, w, h) = obj["box"]
            REFERENCE_POSITIONS[label] = get_center(x, y, w, h)
            detected_names.append(label)
    
    if not detected_names:
        return jsonify({"status": "error", "message": "No se detectaron objetos para vigilar."})

    SYSTEM_ARMED = True
    msg = f"Sistema ARMADO. Vigilando: {', '.join(detected_names)}"
    ALERTS_LOG.append(msg)
    return jsonify({"status": "armed", "objects": detected_names})

@app.route('/disarm_system')
def disarm_system():
    global SYSTEM_ARMED
    SYSTEM_ARMED = False
    return jsonify({"status": "disarmed"})

@app.route('/get_alerts')
def get_alerts():
    return jsonify(ALERTS_LOG[-10:])

# Rutas de Servo (Igual que en tu server.py)
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
    # Usamos puerto 5002 para diferenciarlo
    app.run(host='0.0.0.0', port=5002, debug=False)