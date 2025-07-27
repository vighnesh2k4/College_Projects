import React, { useState } from 'react';
import axios from 'axios';
import ConfirmationModal from './ConfirmationModal';

const StudentList = ({ students, fetchStudents, onEditStudent }) => {
    const [showModal, setShowModal] = useState(false);
    const [studentToDeleteRollNumber, setStudentToDeleteRollNumber] = useState(null);

    const API_URL = 'http://localhost:8000/students/';

    const handleDeleteClick = (rollNumber) => {
        setStudentToDeleteRollNumber(rollNumber);
        setShowModal(true);
    };

    const confirmDelete = async () => {
        try {
            await axios.delete(`${API_URL}${studentToDeleteRollNumber}`);
            alert('Student deleted successfully!');
            fetchStudents();
        } catch (error) {
            console.error('Error deleting student:', error);
            alert('Failed to delete student.');
        } finally {
            setShowModal(false);
            setStudentToDeleteRollNumber(null);
        }
    };

    const cancelDelete = () => {
        setShowModal(false);
        setStudentToDeleteRollNumber(null);
    };

    return (
        <div>
            <h2>Student List</h2>
            {students.length === 0 ? (
                <p>No students found. Add a new student using the form above!</p>
            ) : (
                <ul className="student-list">
                    {students.map((student) => (
                        <li key={student.roll_number}>
                            <div className="student-details">
                                <strong>Roll No: {student.roll_number} - {student.name}</strong> <br />
                                DOB: {student.dob} <br />
                                Branch: {student.branch} <br />
                                Address: {student.address} <br />
                                Mobile: {student.mobile} <br />
                                Gmail: {student.gmail}
                            </div>
                            <div className="student-actions">
                                <button onClick={() => onEditStudent(student)}>Edit</button>
                                <button
                                    className="delete-button"
                                    onClick={() => handleDeleteClick(student.roll_number)}
                                >
                                    Delete
                                </button>
                            </div>
                        </li>
                    ))}
                </ul>
            )}

            {showModal && (
                <ConfirmationModal
                    message="Are you sure you want to delete this student?"
                    onConfirm={confirmDelete}
                    onCancel={cancelDelete}
                />
            )}
        </div>
    );
};

export default StudentList;