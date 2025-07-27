import cv2
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from tensorflow import keras
import numpy as np
import json
import shutil
from tensorflow.keras.models import load_model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import time 
def capture_personalized_gestures(user_id):
    user_data_file = "user_data.csv"
    if not os.path.exists(user_data_file):
        print("User data file not found. Please capture user faces first.")
        return

    user_data = pd.read_csv(user_data_file).set_index("UserID")
    if user_id not in user_data.index:
        print(f"User ID {user_id} not found in the system. Please ensure the user is registered.")
        return

    user_name = user_data.loc[user_id]["UserName"]
    print(f"\n--- Capturing Gestures for User: {user_name} (ID: {user_id}) ---")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    gesture_folder = os.path.join("user_gestures", f"user_{user_id}")
    os.makedirs(gesture_folder, exist_ok=True)

  
    gesture_classes = ["Turn_On_Light", "Turn_Off_Light", "Turn_On_Fan", "Turn_Off_Fan"]

    print("\nInstructions:")
    print("1. A green bounding box will appear on the screen.")
    print("2. Position your hand clearly inside this box for each gesture.")
    print("3. Press 's' to **START** capturing images for the current gesture.")
    print("4. Perform the gesture with slight variations (e.g., slightly different angles, distances, hand positions) while capturing.")
    print("5. Once 200 images are captured, press 'n' to move to the next gesture.")
    print("6. Press 'q' at any time to quit and cancel gesture capture.")

    REQUIRED_GESTURE_IMAGES = 200

    for gesture in gesture_classes:
        gesture_path = os.path.join(gesture_folder, gesture)
        os.makedirs(gesture_path, exist_ok=True)

        print(f"\n--- Prepare to show gesture for: '{gesture.replace('_', ' ')}' ---")
        print(f"Current image count: 0/{REQUIRED_GESTURE_IMAGES}")
        count = 0
        capturing = False

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image.")
                break

            
            roi_x, roi_y, roi_w, roi_h = 100, 100, 300, 300 
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
            roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

            if roi.shape[0] == 0 or roi.shape[1] == 0:
                cv2.putText(frame, "ROI is empty. Adjust camera or frame size.", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow(f"Gesture Capture - {gesture.replace('_', ' ')}", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Gesture capture canceled.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                continue


            if capturing and count < REQUIRED_GESTURE_IMAGES:
                img_path = os.path.join(gesture_path, f"{gesture}_{count}.jpg")
                cv2.imwrite(img_path, roi) 
                count += 1

            cv2.putText(frame, f"Gesture: '{gesture.replace('_', ' ')}'", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Captured: {count}/{REQUIRED_GESTURE_IMAGES} images", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            if capturing:
                cv2.putText(frame, "CAPTURING...", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Press 's' to start, 'n' to next, 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


            cv2.imshow(f"Gesture Capture - {gesture.replace('_', ' ')}", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s') and not capturing:
                capturing = True
                print("Started capturing...")
            elif key == ord('n') and count >= REQUIRED_GESTURE_IMAGES:
                print(f"Completed gesture for '{gesture.replace('_', ' ')}'. Moving to the next gesture.")
                break
            elif key == ord('q'):
                print("Gesture capture canceled.")
                cap.release()
                cv2.destroyAllWindows()
                return

            if capturing and count >= REQUIRED_GESTURE_IMAGES:
                print(f"Captured all {REQUIRED_GESTURE_IMAGES} images for '{gesture.replace('_', ' ')}'. Press 'n' to move to the next gesture.")
                capturing = False 

        if count < REQUIRED_GESTURE_IMAGES:
            print(f"Only {count} images captured for '{gesture.replace('_', ' ')}'. This may affect model accuracy.")

    cap.release()
    cv2.destroyAllWindows()
    print("Gesture capture completed for all gestures.")


def train_gesture_model(user_id, gesture_model_path):
    gesture_folder = os.path.join("user_gestures", f"user_{user_id}")
    if not os.path.exists(gesture_folder) or not os.listdir(gesture_folder):
        print(f"No gesture data found for user {user_id}. Please capture personalized gestures first.")
        return

    
    gesture_classes = ["Turn_On_Light", "Turn_Off_Light", "Turn_On_Fan", "Turn_Off_Fan"] 
    has_data = False
    for gesture in gesture_classes:
        if os.path.exists(os.path.join(gesture_folder, gesture)) and len(os.listdir(os.path.join(gesture_folder, gesture))) > 0:
            has_data = True
            break
    if not has_data:
        print(f"No gesture image data found in '{gesture_folder}'. Please ensure images were captured correctly.")
        return


    print("\n--- Training Gesture Recognition Model ---")
    print("Loading and augmenting image data...")

    data_gen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2, 
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    IMG_HEIGHT, IMG_WIDTH = 64, 64

    train_data = data_gen.flow_from_directory(
        gesture_folder,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode="categorical",
        subset="training",
    )
    val_data = data_gen.flow_from_directory(
        gesture_folder,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=32,
        class_mode="categorical",
        subset="validation",
    )

    if train_data.samples == 0:
        print("No training images found. Please ensure gestures were captured correctly in 'capture_personalized_gestures'.")
        return
    if val_data.samples == 0:
        print("No validation images found. This might happen if 'validation_split' is too high for limited data.")
        print("Consider capturing more data or adjusting 'validation_split'.")
        return

   
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(train_data.class_indices), activation="softmax"),
    ])

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1)

    print(f"Starting training for User ID: {user_id}...")
    try:
        history = model.fit(
            train_data,
            validation_data=val_data,
            epochs=50,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        print("Training history (last 5 epochs):")
        print(pd.DataFrame(history.history).tail())

        model.save(gesture_model_path)
        with open(gesture_model_path.replace(".h5", "_class_indices.json"), "w") as f:
            json.dump(train_data.class_indices, f)
        print(f"Training completed. Gesture model saved to {gesture_model_path}.")
    except Exception as e:
        print(f"An error occurred during model training: {e}")
        print("Please check your data and ensure TensorFlow is correctly installed.")


def recognize_gesture(gesture_model_path, send_command_func, labels):
    

    class_indices_path = gesture_model_path.replace(".h5", "_class_indices.json")
    if not os.path.exists(class_indices_path):
        print(f"Error: Class indices file not found: {class_indices_path}")
        print("Please train the gesture model first to create this file.")
        return

    if not os.path.exists(gesture_model_path):
        print(f"Error: Gesture model file not found: {gesture_model_path}")
        print("Please train the gesture model first.")
        return

    try:
        model = load_model(gesture_model_path)
    except Exception as e:
        print(f"Error loading gesture model from {gesture_model_path}. It might be corrupted. Please retrain. Error: {e}")
        return

    

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("\n--- Gesture Recognition ---")
    print("Perform your gestures within the green bounding box. Press 'q' to quit.")

   
    prediction_history = []
    HISTORY_LENGTH = 10 
    CONFIDENCE_THRESHOLD = 0.8 
    LAST_COMMAND_SENT_TIME = 0
    COMMAND_COOLDOWN = 3 
    gesture_to_esp_command = {
        "Turn_On_Light": "light/on",
        "Turn_Off_Light": "light/off",
        "Turn_On_Fan": "fan/on",
        "Turn_Off_Fan": "fan/off",
       
    }


    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        roi_x, roi_y, roi_w, roi_h = 100, 100, 300, 300
        cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (0, 255, 0), 2)
        roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

        if roi.shape[0] == 0 or roi.shape[1] == 0:
            cv2.putText(frame, "Please place hand in the green box.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Gesture Recognition", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            continue

        resized_roi = cv2.resize(roi, (64, 64)) 
        input_array = np.expand_dims(resized_roi / 255.0, axis=0) 

        try:
            predictions = model.predict(input_array, verbose=0)[0] 
            prediction_history.append(predictions)
            if len(prediction_history) > HISTORY_LENGTH:
                prediction_history.pop(0)

            
            avg_predictions = np.mean(prediction_history, axis=0)
            predicted_label_index = np.argmax(avg_predictions)
            predicted_gesture_name_raw = labels[predicted_label_index] 
            confidence = np.max(avg_predictions)

            display_text = "Recognizing..."
            if confidence > CONFIDENCE_THRESHOLD:
                
                predicted_gesture_display = predicted_gesture_name_raw.replace('_', ' ')
                display_text = f"{predicted_gesture_display} ({confidence:.2f})"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

               
                if (time.time() - LAST_COMMAND_SENT_TIME) > COMMAND_COOLDOWN:
                    esp_command = gesture_to_esp_command.get(predicted_gesture_name_raw)
                    if esp_command:
                        print(f"Recognized '{predicted_gesture_display}'. Sending command: {esp_command}")
                        send_command_func(esp_command) 
                        LAST_COMMAND_SENT_TIME = time.time() 
                    else:
                        print(f"Warning: No ESP command mapping for gesture '{predicted_gesture_name_raw}'.")
            else:
                display_text = f"Gesture not recognized (Confidence: {confidence:.2f})"
                cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        except Exception as e:
            cv2.putText(frame, f"Prediction error: {e}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)


        cv2.imshow("Gesture Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Gesture recognition stopped.")