import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from typing import List, Optional
from datetime import date

from sqlalchemy import create_engine, Column, Integer, String, Date, Text
from sqlalchemy.orm import sessionmaker, declarative_base, Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
EMAIL_SENDER_ADDRESS = os.getenv("EMAIL_SENDER_ADDRESS")
EMAIL_SENDER_PASSWORD = os.getenv("EMAIL_SENDER_PASSWORD")

app = FastAPI(
    title="Student Management API",
    description="API for managing student records with email notifications.",
    version="1.0.0"
)

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class Student(Base):
    __tablename__ = "students"

    roll_number = Column(Integer, primary_key=True)
    name = Column(String)
    dob = Column(Date)
    branch = Column(String)
    address = Column(Text)
    mobile = Column(String)
    gmail = Column(String, unique=True)

class StudentBase(BaseModel):
    name: str
    dob: date
    branch: str
    address: str
    mobile: str
    gmail: EmailStr

class StudentCreate(StudentBase):
    pass

class StudentUpdate(BaseModel):
    name: Optional[str] = None
    dob: Optional[date] = None
    branch: Optional[str] = None
    address: Optional[str] = None
    mobile: Optional[str] = None
    gmail: Optional[EmailStr] = None

class StudentInDB(StudentBase):
    roll_number: int

    class Config:
        from_attributes = True

def send_email(to_email: str, subject: str, body: str):
    if not EMAIL_SENDER_ADDRESS or not EMAIL_SENDER_PASSWORD:
        print("Email sender credentials not configured. Skipping email.")
        return

    msg = MIMEMultipart()
    msg['From'] = EMAIL_SENDER_ADDRESS
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_SENDER_ADDRESS, EMAIL_SENDER_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER_ADDRESS, to_email, text)
        server.quit()
        print(f"Email sent to {to_email}")
    except Exception as e:
        print(f"Failed to send email to {to_email}: {e}")

@app.get("/students/", response_model=List[StudentInDB], tags=["Students"])
def read_students(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    students = db.query(Student).offset(skip).limit(limit).all()
    return students

@app.post("/students/", response_model=StudentInDB, status_code=status.HTTP_201_CREATED, tags=["Students"])
def create_student(student: StudentCreate, db: Session = Depends(get_db)):
    db_student = db.query(Student).filter(Student.gmail == student.gmail).first()
    if db_student:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Student with this gmail already exists"
        )
    
    db_student = Student(**student.model_dump())
    db.add(db_student)
    db.commit()
    db.refresh(db_student)

    subject = "Student Record Created"
    body = f"A new student record has been created:\n\nRoll Number: {db_student.roll_number}\nName: {db_student.name}\nBranch: {db_student.branch}\nMobile: {db_student.mobile}\nGmail: {db_student.gmail}"
    send_email(db_student.gmail, subject, body)
    send_email(EMAIL_SENDER_ADDRESS, subject, body)

    return db_student

@app.get("/students/{roll_number}", response_model=StudentInDB, tags=["Students"])
def read_student(roll_number: int, db: Session = Depends(get_db)):
    student = db.query(Student).filter(Student.roll_number == roll_number).first()
    if student is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    return student

@app.put("/students/{roll_number}", response_model=StudentInDB, tags=["Students"])
def update_student(roll_number: int, student: StudentUpdate, db: Session = Depends(get_db)):
    db_student = db.query(Student).filter(Student.roll_number == roll_number).first()
    if db_student is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")

    update_data = student.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_student, key, value)

    db.add(db_student)
    db.commit()
    db.refresh(db_student)

    subject = "Student Record Updated"
    body = f"The student record for Roll Number {db_student.roll_number} has been updated:\n\nName: {db_student.name}\nBranch: {db_student.branch}\nMobile: {db_student.mobile}\nGmail: {db_student.gmail}"
    send_email(db_student.gmail, subject, body)
    send_email(EMAIL_SENDER_ADDRESS, subject, body)

    return db_student

@app.delete("/students/{roll_number}", status_code=status.HTTP_204_NO_CONTENT, tags=["Students"])
def delete_student(roll_number: int, db: Session = Depends(get_db)):
    db_student = db.query(Student).filter(Student.roll_number == roll_number).first()
    if db_student is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student not found")
    
    deleted_student_gmail = db_student.gmail
    deleted_student_name = db_student.name

    db.delete(db_student)
    db.commit()

    subject = "Student Record Deleted"
    body = f"The student record for {deleted_student_name} (Roll Number {roll_number}) has been deleted."
    send_email(deleted_student_gmail, subject, body)
    send_email(EMAIL_SENDER_ADDRESS, subject, body)

    return