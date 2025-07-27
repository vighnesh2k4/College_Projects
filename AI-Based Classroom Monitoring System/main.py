from capture_image import capture_faces
from train_image import train_images
from entry_recognition import entry_recognition
from exit_recognition import exit_recognition
from attendance import view_student_attendance, view_today_attendance, view_weekly_attendance

def main_menu():
    while True:
        print("""
        **********************************************
        ***** Face Recognition Attendance System *****
        **********************************************

         ********** WELCOME MENU **********
        [1] Capture Faces
        [2] Train Images
        [3] Entry Point Recognition
        [4] Exit Point Recognition
        [5] View Specific Student Attendance
        [6] Today's Overall Attendance
        [7] Weekly Attendance Report
        [8] Quit
        """)
        choice = input("Enter Choice: ").strip()
        try:
            if choice == "1":
                capture_faces()
            elif choice == "2":
                train_images()
            elif choice == "3":
                entry_recognition()
            elif choice == "4":
                exit_recognition()
            elif choice == "5":
                student_id = input("Enter Student ID: ").strip()
                view_student_attendance(student_id)
            elif choice == "6":
                view_today_attendance()
            elif choice == "7":
                view_weekly_attendance()
            elif choice == "8":
                print("Exiting. Thank you!")
                break
            else:
                print("Invalid input! Please enter a number between 1-8.")
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main_menu()