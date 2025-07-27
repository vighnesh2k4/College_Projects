import cv2
import pandas as pd
from datetime import datetime

def exit_recognition():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainer.yml")
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    try:
        users = pd.read_csv("UserState.csv", dtype={"Id": str})
        if users.empty:
            users = pd.DataFrame(columns=["Id", "State"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        users = pd.DataFrame(columns=["Id", "State"])
        users.to_csv("UserState.csv", index=False)

    cam = cv2.VideoCapture(0)
    print("Press 'q' to quit camera.")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to open camera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            Id, confidence = recognizer.predict(roi_gray)
            if confidence < 50:
                Id_str = str(Id)
                if ((users["Id"] == Id_str) & (users["State"] == "inside")).any():
                    print(f"Exit recorded for ID: {Id}")
                    cv2.putText(frame, f"ID: {Id}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    users.loc[users["Id"] == Id_str, "State"] = "outside"
                    users.to_csv("UserState.csv", index=False)
                    current_time = datetime.now()
                    day = current_time.strftime("%A")
                    date = current_time.strftime("%Y-%m-%d")
                    time = current_time.strftime("%H:%M:%S")
                    with open("ExitLog.csv", "a") as f:
                        f.write(f"{Id},{date},{day},{time},exit \n")
            else:
                continue

        cv2.imshow("Exit Recognition", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()
