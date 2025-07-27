import cv2
import os
import pandas as pd

def capture_faces():
    if not os.path.exists("Dataset"):
        os.makedirs("Dataset")
        print("Created 'Dataset' folder.")

    try:
        students = pd.read_csv("StudentDetails.csv")
        if students.empty:
            students = pd.DataFrame(columns=["Id", "Name"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        students = pd.DataFrame(columns=["Id", "Name"])
        students.to_csv("StudentDetails.csv", index=False)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    cam = cv2.VideoCapture(0)
    user_id = input("Enter Your ID: ").strip()
    user_name = input("Enter Your Name: ").strip()

    existing_ids = pd.read_csv("StudentDetails.csv")["Id"].astype(str).tolist()
    if user_id in existing_ids:
        print("Error: ID already exists. Please use a unique ID.")
        cam.release()
        cv2.destroyAllWindows()
        return

    count = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y + h, x:x + w]
            cv2.imwrite(f"Dataset/User.{user_id}.{count}.jpg", face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Capturing Faces", frame)
        if cv2.waitKey(100) & 0xFF == ord('q') or count >= 20:
            break

    cam.release()
    cv2.destroyAllWindows()
    with open("StudentDetails.csv", "a") as f:
        f.write(f"{user_id},{user_name}\n")
    print(f"Images saved for ID: {user_id}, Name: {user_name}")