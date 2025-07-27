import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import StudentList from './components/StudentList';
import StudentForm from './components/StudentForm';

function App() {
    const [students, setStudents] = useState([]);
    const [studentToEdit, setStudentToEdit] = useState(null);

    const API_URL = 'http://localhost:8000/students/';

    const fetchStudents = useCallback(async () => {
        try {
            const response = await axios.get(API_URL);
            setStudents(response.data);
        } catch (error) {
            console.error('Error fetching students:', error);
        }
    }, []);

    useEffect(() => {
        fetchStudents();
    }, [fetchStudents]);

    const handleStudentAdded = (newStudent) => {
        setStudents((prevStudents) => [...prevStudents, newStudent]);
        setStudentToEdit(null);
        fetchStudents();
    };

    const handleStudentUpdated = (updatedStudent) => {
        setStudents((prevStudents) =>
            prevStudents.map((s) => (s.roll_number === updatedStudent.roll_number ? updatedStudent : s))
        );
        setStudentToEdit(null);
        fetchStudents();
    };

    const handleEditStudent = (student) => {
        setStudentToEdit(student);
    };

    return (
        <div className="App">
            <h1>Student Management System</h1>
            <StudentForm
                studentToEdit={studentToEdit}
                onStudentAdded={handleStudentAdded}
                onStudentUpdated={handleStudentUpdated}
            />
            <StudentList
                students={students}
                fetchStudents={fetchStudents}
                onEditStudent={handleEditStudent}
            />
        </div>
    );
}

export default App;