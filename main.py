import cv2
import numpy as np
import time
from database import init_db, get_students, mark_attendance, generate_excel

# Inicialización de la base de datos
init_db()

def load_labels(filename='labels.txt'):
    with open(filename, 'r') as f:
        labels = {}
        for line in f.readlines():
            label, name = line.strip().split(':')
            labels[int(label)] = name
    return labels

def run_recognition():
    # Cargar modelo de reconocimiento facial de OpenCV
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_model.xml')
    labels = load_labels()

    # Intentar con diferentes backends de captura de video
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_VFW, cv2.CAP_ANY]
    video_capture = None
    for backend in backends:
        video_capture = cv2.VideoCapture(1, backend)
        if video_capture.isOpened():
            print(f"Using backend: {backend}")
            break

    if not video_capture or not video_capture.isOpened():
        print("No se puede acceder a la cámara con ninguno de los backends disponibles.")
        return

    last_marked = {}
    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("No se puede capturar el frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        current_time = time.time()
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face)

            if confidence < 50:  # Ajusta este valor según sea necesario
                student_name = labels.get(label, "Desconocido")
                if student_name != "Desconocido":
                    last_marked_time = last_marked.get(label, 0)
                    if current_time - last_marked_time > 60:
                        mark_attendance(label + 1)  # Ajuste para índices de etiquetas
                        last_marked[label] = current_time
                        print(f"Asistencia marcada para {student_name} a las {time.ctime(current_time)}")
                
                cv2.putText(frame, student_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        cv2.imshow('Reconocimiento Facial', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Llamar a las funciones según se necesite
    view_students()
    view_attendance()
    run_recognition()
    generate_excel()
