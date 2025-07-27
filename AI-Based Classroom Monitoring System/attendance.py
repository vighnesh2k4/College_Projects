import csv
from datetime import datetime

# File paths
ENTRY_FILE = "EntryLog.csv"
EXIT_FILE = "ExitLog.csv"
STUDENT_FILE = "StudentDetails.csv"

# Predefined periods (24-hour format)
PERIODS = {
    "9:00 - 10:00": (9, 10),
    "10:00 - 11:00": (10, 11),
    "11:00 - 12:00": (11, 12),
    "12:00 - 1:00": (12, 13),
    "2:00 - 3:00": (14, 15),
    "3:00 - 4:00": (15, 16),
}

def get_student_name(student_id):
    """Fetch the student name by their ID from the student details file."""
    with open(STUDENT_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Id"] == student_id:
                return row["Name"]
    return "Unknown"

def calculate_duration(entry_time, exit_time):
    """Calculate the duration (in minutes) between entry and exit times."""
    entry = datetime.strptime(entry_time, "%H:%M:%S")
    exit = datetime.strptime(exit_time, "%H:%M:%S")
    return (exit - entry).total_seconds() / 60

def get_periods(entry_times, exit_times):
    """Map entry and exit times to predefined periods, considering a minimum 45-minute presence."""
    attended_periods = []
    for entry, exit in zip(entry_times, exit_times):
        duration = calculate_duration(entry, exit)
        if duration >= 45:
            entry_hour = int(entry.split(":")[0])
            exit_hour = int(exit.split(":")[0])
            for period, (start, end) in PERIODS.items():
                if start <= entry_hour < end or start < exit_hour <= end:
                    attended_periods.append(period)
    return list(set(attended_periods))  # Remove duplicates

def view_student_attendance(student_id):
    """View the attendance details of a specific student, including time periods for all dates."""
    student_name = get_student_name(student_id)
    attendance = {}

    # Read entries and exits for the student
    with open(ENTRY_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == student_id:
                date = row[1]
                time = row[3]
                if date not in attendance:
                    attendance[date] = {"entries": [], "exits": []}
                attendance[date]["entries"].append(time)

    with open(EXIT_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == student_id:
                date = row[1]
                time = row[3]
                if date in attendance:
                    attendance[date]["exits"].append(time)

    # Display attendance details
    print(f"\n--- Attendance Details for ID {student_id} ({student_name}) ---")
    for date, times in sorted(attendance.items()):
        day = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
        print(f"Date: {date} ({day})")
        entry_times = times["entries"]
        exit_times = times["exits"]
        if not entry_times or not exit_times:
            print("  No entry/exit records found.")
            continue

        print("  Entries/Exits:")
        for entry, exit in zip(entry_times, exit_times):
            print(f"   - Entry: {entry}")
            print(f"   - Exit: {exit}")

        attended_periods = get_periods(entry_times, exit_times)
        print(f"  Periods Attended: {len(attended_periods)}")
        print(f"  Present in Periods: {', '.join(attended_periods) if attended_periods else 'N/A'}")

def view_today_attendance():
    """
    View today's overall attendance for all students.
    """
    today_date = datetime.now().strftime("%Y-%m-%d")
    attendance_summary = {}

    # Process today's entries and exits
    with open(ENTRY_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == today_date:
                student_id = row[0]
                if student_id not in attendance_summary:
                    attendance_summary[student_id] = {"entries": [], "exits": []}
                attendance_summary[student_id]["entries"].append(row[3])

    with open(EXIT_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[1] == today_date:
                student_id = row[0]
                if student_id in attendance_summary:
                    attendance_summary[student_id]["exits"].append(row[3])

    print("\n--- Today's Overall Attendance ---")
    for student_id, times in attendance_summary.items():
        student_name = get_student_name(student_id)
        first_in = min(times["entries"], default="N/A")
        last_out = max(times["exits"], default="N/A")
        print(f"ID: {student_id} ({student_name}), First In: {first_in}, Last Out: {last_out}")

def view_weekly_attendance():
    """
    View the weekly attendance for all students.
    """
    weekly_summary = {}

    # Process entries for the week
    with open(ENTRY_FILE, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            student_id = row[0]
            date = row[1]
            day = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
            if student_id not in weekly_summary:
                weekly_summary[student_id] = {}
            weekly_summary[student_id][day] = True  # Mark presence

    print("\n--- Weekly Attendance ---")
    print(f"{'ID':<10}{'Name':<15}{'Monday':<10}{'Tuesday':<10}{'Wednesday':<10}{'Thursday':<10}{'Friday':<10}{'Saturday':<10}")
    for student_id, days in weekly_summary.items():
        student_name = get_student_name(student_id)
        attendance_row = [student_id, student_name]
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
            attendance_row.append('P' if days.get(day) else 'A')
        print(f"{attendance_row[0]:<10}{attendance_row[1]:<15}{attendance_row[2]:<10}{attendance_row[3]:<10}{attendance_row[4]:<10}{attendance_row[5]:<10}{attendance_row[6]:<10}{attendance_row[7]:<10}")
