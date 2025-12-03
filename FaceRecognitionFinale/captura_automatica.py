# === INICIO: Código para captura_automatica.py ===
import cv2
import os
import sys
import time
from datetime import datetime
from picamera2 import Picamera2

def create_folder(name):
    dataset_folder = "dataset"
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    person_folder = os.path.join(dataset_folder, name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)
    return person_folder

def capture_photos(name, num_photos=10):
    folder = create_folder(name)

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()

    print("Iniciando cámara... 3 segundos para prepararse.")
    time.sleep(3) # Tiempo para que la cámara se estabilice

    photo_count = 0
    print(f"Tomando {num_photos} fotos para {name}...")

    for i in range(num_photos):
        frame = picam2.capture_array()

        photo_count += 1
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)

        # Guardamos el frame (la captura de la cámara)
        cv2.imwrite(filepath, frame) 

        print(f"Foto {photo_count}/{num_photos} guardada: {filepath}")
        time.sleep(1) # Espera 1 segundo entre fotos

    picam2.stop()
    print(f"Captura completada para {name}.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Error: Debes pasar un nombre como argumento.")
        sys.exit(1)

    PERSON_NAME = sys.argv[1]
    capture_photos(PERSON_NAME)
# === FIN: Código para captura_automatica.py ===