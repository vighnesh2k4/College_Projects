import os
import shutil
import json 
import pandas as pd 

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  

from facial_recognition import capture_user_faces, train_faces, authenticate_user
from gesture_recognition import capture_personalized_gestures, train_gesture_model, recognize_gesture
from esp_commander import send_command, direct_command_mode 

def main():
    print("\n--- Welcome to Smart Home System ---")

    
    user_id = None
    user_name = None

    
    while user_id is None:
        print("\n1. Register New User\n2. Authenticate Existing User\n3. Reset System (Start Fresh)\n4. Exit")
        option = input("Choose an option: ")
        if option == "1":
            user_id, user_name = capture_user_faces() 
            if user_id: 
                print(f"User registered successfully: {user_name} (ID: {user_id})")
            else:
                print("User registration failed. Please try again.")
        elif option == "2":
            user_name, user_id = authenticate_user()  
            if user_name:
                print(f"Welcome back, {user_name} (ID: {user_id})!")
            else:
                print("Authentication failed. Please try again.")
        elif option == "3":
            confirm = input("Are you sure you want to reset the system? This will delete all data (yes/no): ").strip().lower()
            if confirm == "yes":
                directories_to_remove = ["user_faces", "user_gestures"]
                files_to_remove = ["face_model.yml", "user_data.csv"]
                for directory in directories_to_remove:
                    if os.path.exists(directory):
                        shutil.rmtree(directory)
                        print(f"Removed directory: {directory}")
                for file in files_to_remove:
                    if os.path.exists(file):
                        os.remove(file)
                        print(f"Removed file: {file}")
                
                for file in os.listdir():
                    if file.startswith("gesture_model_") and (file.endswith(".h5") or file.endswith("_class_indices.json")):
                        os.remove(file)
                        print(f"Removed file: {file}")
                print("System reset completed. All data has been cleared.")
               
                user_id = None
                user_name = None
            else:
                print("System reset canceled.")
        elif option == "4":
            print("Goodbye!")
            return 
        else:
            print("Invalid option. Please try again.")

   
    while True:
        print("\n--- Smart Home System ---")
        print("1. Train Facial Recognition Model")
        print("2. Capture Personalized Gestures")
        print("3. Train Gesture Recognition Model")
        print("4. Use Personalized Gestures for Control")
        print("5. Use Direct Commands for Control")
        print("6. Exit")
        choice = input("Enter your choice: ")

        if choice == "1":
            train_faces()
        elif choice == "2":
           
            capture_personalized_gestures(user_id)
        elif choice == "3":
            gesture_model_path = f"gesture_model_{user_id}.h5"
            train_gesture_model(user_id, gesture_model_path)
        elif choice == "4":
           
            gesture_model_path = f"gesture_model_{user_id}.h5"
            class_indices_path = gesture_model_path.replace(".h5", "_class_indices.json")

            if not os.path.exists(gesture_model_path) or not os.path.exists(class_indices_path):
                print(f"No gesture model or class indices found for user {user_name}. Please capture and train gestures first.")
                continue

            with open(class_indices_path) as f:
                class_indices = json.load(f)
            labels = {v: k for k, v in class_indices.items()} 

            print("\n--- Gesture Control Mode ---")
            print("Perform your gestures. Press 'q' to quit.")
            
            recognize_gesture(gesture_model_path, send_command, labels) 
        elif choice == "5": 
            direct_command_mode()
        elif choice == "6": 
            print("Exiting Smart Home System. Goodbye!")
            break 
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()