from flask import Flask, render_template, Response
import cv2
import pickle
import datetime
import os

# Cargar nombres conocidos
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_names = data["names"]

# Lista de correos (del archivo principal)
from codigo_nuevo import EMAIL_RECEIVERS

app = Flask(__name__)

# --- STREAM DE CAMARA (solo video crudo, sin reconocimiento) ---
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ok, frame = camera.read()
        if not ok:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# PÃ¡gina principal
@app.route('/')
def index():
    # Ultimo evento registrado desde la carpeta de desconocidos o conocidos
    folder_known = "capturas_conocidos"
    folder_unknown = "capturas_desconocidos"

    def last_file(folder):
        files = [f for f in os.listdir(folder)] if os.path.exists(folder) else []
        if not files:
            return "Sin registros"
        files.sort()
        return files[-1]

    ultimo_conocido = last_file(folder_known)
    ultimo_desconocido = last_file(folder_unknown)

    return render_template("dashboard.html",
                           known_list=known_face_names,
                           correos=EMAIL_RECEIVERS,
                           conocido=ultimo_conocido,
                           desconocido=ultimo_desconocido)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
