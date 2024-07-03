import cv2
import os
import numpy as np

def augment_image(image):
    augmented_images = []
    rows, cols = image.shape[:2]

    # Rotación
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))
    augmented_images.append(rotated)

    # Ajuste de brillo
    bright = cv2.convertScaleAbs(image, alpha=1.2, beta=30)
    augmented_images.append(bright)

    # Recorte
    cropped = image[10:rows-10, 10:cols-10]
    cropped = cv2.resize(cropped, (cols, rows))
    augmented_images.append(cropped)

    # Volteo horizontal
    flipped = cv2.flip(image, 1)
    augmented_images.append(flipped)

    return augmented_images

def normalize_image(image):
    return cv2.equalizeHist(image)

def load_images(base_dir):
    faces = []
    labels = []
    label_to_name = {}

    for label, name in enumerate(os.listdir(base_dir)):
        label_path = os.path.join(base_dir, name)
        if not os.path.isdir(label_path):
            continue
        label_to_name[label] = name
        for image_name in os.listdir(label_path):
            image_path = os.path.join(label_path, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = normalize_image(image)
                faces.append(image)
                labels.append(label)
                augmented_images = augment_image(image)
                for aug_image in augmented_images:
                    faces.append(aug_image)
                    labels.append(label)
    return faces, labels, label_to_name

def train_model():
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    base_dir = 'dataset'  # Carpeta que contiene subcarpetas de imágenes de cada persona
    faces, labels, label_to_name = load_images(base_dir)

    face_recognizer.train(faces, np.array(labels))
    face_recognizer.write('face_model.xml')

    # Guardar las etiquetas y nombres en un archivo
    with open('labels.txt', 'w') as f:
        for label, name in label_to_name.items():
            f.write(f"{label}:{name}\n")

    print("Modelo entrenado y guardado correctamente.")

if __name__ == "__main__":
    train_model()
