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
camera = cv2.VideoCapture(0)

def generar_video():
    while True:
        ret, frame = camera.read()
        if not ret:
            continue
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

@app.route('/agregar_rostro', methods=['POST'])
def agregar_rostro():
    nombre = request.form.get("nombre")
    if not nombre:
        return jsonify({"error": "Debes enviar un nombre"}), 400

    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.loads(f.read())
    else:
        data = {"encodings": [], "names": []}

    # Por ahora solo guardamos nombre, en tu codigo real agregar encodings
    data["names"].append(nombre)

    with open(ENCODINGS_FILE, "wb") as f:
        f.write(pickle.dumps(data))

    return jsonify({"success": True})

@app.route('/eliminar_rostro', methods=['POST'])
def eliminar_rostro():
    nombre = request.form.get("nombre")
    if not nombre:
        return jsonify({"error": "Debes enviar un nombre"}), 400

    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.loads(f.read())
        new_names = []
        new_encodings = []
        for n, e in zip(data["names"], data.get("encodings", [])):
            if n != nombre:
                new_names.append(n)
                new_encodings.append(e)
        data["names"] = new_names
        data["encodings"] = new_encodings
        with open(ENCODINGS_FILE, "wb") as f:
            f.write(pickle.dumps(data))
    return jsonify({"success": True})

@app.route('/agregar_correo', methods=['POST'])
def agregar_correo():
    correo = request.form.get("correo")
    if correo and correo not in EMAIL_RECEIVERS:
        EMAIL_RECEIVERS.append(correo)
    return jsonify({"success": True})

@app.route('/borrar_correo', methods=['POST'])
def borrar_correo():
    correo = request.form.get("correo")
    if correo in EMAIL_RECEIVERS:
        EMAIL_RECEIVERS.remove(correo)
    return jsonify({"success": True})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
