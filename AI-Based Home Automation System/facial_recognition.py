import cv2
import os
import pandas as pd
import numpy as np
import shutil 

USER_DATA_FILE = "user_data.csv"
FACE_MODEL_FILE = "face_model.yml"
FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

def capture_user_faces():
    
    os.makedirs("user_faces", exist_ok=True)

    if os.path.exists(USER_DATA_FILE):
        user_data = pd.read_csv(USER_DATA_FILE)
    else:
        user_data = pd.DataFrame(columns=["UserID", "UserName"])

    user_id = None
    user_name = None

    while True:
        try:
            user_id_input = input("Enter a new User ID (integer): ")
            user_id = int(user_id_input)
        except ValueError:
            print("User ID must be an integer. Please try again.")
            continue

        if user_id in user_data["UserID"].values:
            print(f"User ID {user_id} is already in use. Please choose a different ID.")
        else:
            break

    user_name = input("Enter User Name: ")

    new_user_row = pd.DataFrame({"UserID": [user_id], "UserName": [user_name]})
    user_data = pd.concat([user_data, new_user_row], ignore_index=True)
    user_data.to_csv(USER_DATA_FILE, index=False)

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {FACE_CASCADE_PATH}")
        return None, None

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None

    user_folder = os.path.join("user_faces", f"user_{user_id}")
    os.makedirs(user_folder, exist_ok=True)

    print("\n--- Face Capture for Registration ---")
    print("Keep your face clearly visible in the frame.")
    print("Press 's' to **START** capturing faces. Press 'q' to quit.")
    capturing = False
    img_count = 0
    REQUIRED_IMAGES = 50

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            if capturing and img_count < REQUIRED_IMAGES:
                face_img = gray[y:y + h, x:x + w]
                
                face_img = cv2.resize(face_img, (100, 100))
                img_path = os.path.join(user_folder, f"face_{img_count}.jpg")
                cv2.imwrite(img_path, face_img)
                img_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.putText(frame, f"Captured: {img_count}/{REQUIRED_IMAGES} images", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if capturing:
            cv2.putText(frame, "CAPTURING...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, "Press 's' to start", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


        cv2.imshow("Face Capture", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and not capturing:
            capturing = True
            print("Started capturing faces...")
        elif key == ord('q'):
            print("Face capture canceled.")
            break

        if img_count >= REQUIRED_IMAGES:
            print(f"Captured {REQUIRED_IMAGES} face images.")
            break

    cap.release()
    cv2.destroyAllWindows()

    if img_count >= REQUIRED_IMAGES:
        return user_id, user_name
    else:
        print(f"Not enough images captured ({img_count}/{REQUIRED_IMAGES}). Registration incomplete.")
       
       
        if os.path.exists(user_folder):
            shutil.rmtree(user_folder)
  
        if user_id in user_data["UserID"].values:
            user_data = user_data[user_data["UserID"] != user_id]
            user_data.to_csv(USER_DATA_FILE, index=False)
        return None, None


def train_faces():
    if not os.path.exists("user_faces") or not os.listdir("user_faces"):
        print("No user face data found. Please capture user faces first.")
        return

    if not os.path.exists(USER_DATA_FILE):
        print("User data file not found. Please register users first.")
        return

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    user_data = pd.read_csv(USER_DATA_FILE)

    print("Loading face data for training...")
    for _, row in user_data.iterrows():
        user_id = row["UserID"]
        user_folder = os.path.join("user_faces", f"user_{user_id}")
        if not os.path.exists(user_folder):
            print(f"Warning: Face data for user {user_id} not found at {user_folder}. Skipping.")
            continue
        for img_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(int(user_id))
            else:
                print(f"Warning: Could not read image {img_path}. Skipping.")

    if not faces:
        print("No valid face images found for training. Please capture faces correctly.")
        return

    try:
        print("Training facial recognition model...")
        face_recognizer.train(faces, np.array(labels))
        face_recognizer.save(FACE_MODEL_FILE)
        print(f"Facial recognition model trained and saved to {FACE_MODEL_FILE}.")
    except Exception as e:
        print(f"Error during facial recognition model training: {e}")


def authenticate_user():
    if not os.path.exists(FACE_MODEL_FILE):
        print("Face model not found. Please train the model first (Option 1 in main menu).")
        return None, None

    face_cascade = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_cascade.empty():
        print(f"Error: Could not load face cascade from {FACE_CASCADE_PATH}")
        return None, None

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        face_recognizer.read(FACE_MODEL_FILE)
    except cv2.error as e:
        print(f"Error reading face model file {FACE_MODEL_FILE}. It might be corrupted or empty. Please retrain. Error: {e}")
        return None, None

    if not os.path.exists(USER_DATA_FILE):
        print("User data file not found. Please register users first.")
        return None, None
    user_data = pd.read_csv(USER_DATA_FILE).set_index("UserID")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None, None

    print("\nAuthenticating user... Please look at the camera clearly. Press 'q' to quit.")
    authenticated_count = 0
    REQUIRED_AUTH_SUCCESSES = 10 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        recognized_user_name = "Unknown"
        recognized_user_id = -1
        current_confidence = 0

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            
            face_img = cv2.resize(face_img, (100, 100))

            label, confidence = face_recognizer.predict(face_img)

            if confidence < 70:  
                if label in user_data.index:
                    recognized_user_name = user_data.loc[label]["UserName"]
                    recognized_user_id = label
                    current_confidence = confidence
                    cv2.putText(frame, f"{recognized_user_name} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    authenticated_count += 1
                else:
                    cv2.putText(frame, f"Unknown ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    authenticated_count = 0 
            else:
                cv2.putText(frame, f"Unknown ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                authenticated_count = 0 


        cv2.putText(frame, f"Auth Successes: {authenticated_count}/{REQUIRED_AUTH_SUCCESSES}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Authentication", frame)

        if authenticated_count >= REQUIRED_AUTH_SUCCESSES:
            print(f"User '{recognized_user_name}' (ID: {recognized_user_id}) authenticated successfully!")
            cap.release()
            cv2.destroyAllWindows()
            return recognized_user_name, recognized_user_id

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Authentication canceled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Authentication failed.")
    return None, None