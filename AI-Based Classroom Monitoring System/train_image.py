import cv2
import os
import numpy as np

def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    faces = []
    ids = []

    for image in os.listdir("Dataset"):
        img_path = os.path.join("Dataset", image)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        id_ = int(image.split(".")[1])
        faces.append(img)
        ids.append(id_)

    recognizer.train(faces, np.array(ids))
    recognizer.write("Trainer.yml")
    print("Training completed.")