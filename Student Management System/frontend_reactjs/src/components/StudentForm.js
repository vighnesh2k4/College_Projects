import React, { useState, useEffect } from 'react';
import axios from 'axios';

const StudentForm = ({ studentToEdit, onStudentAdded, onStudentUpdated }) => {
    const [name, setName] = useState('');
    const [dob, setDob] = useState('');
    const [branch, setBranch] = useState('');
    const [address, setAddress] = useState('');
    const [mobile, setMobile] = useState('');
    const [gmail, setGmail] = useState('');
    const [error, setError] = useState('');

    const API_URL = 'http://localhost:8000/students/';

    useEffect(() => {
        if (studentToEdit) {
            setName(studentToEdit.name);
            setDob(studentToEdit.dob);
            setBranch(studentToEdit.branch);
            setAddress(studentToEdit.address);
            setMobile(studentToEdit.mobile);
            setGmail(studentToEdit.gmail);
        } else {
            setName('');
            setDob('');
            setBranch('');
            setAddress('');
            setMobile('');
            setGmail('');
            setError('');
        }
    }, [studentToEdit]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        const studentData = {
            name: name,
            dob: dob,
            branch: branch,
            address: address,
            mobile: mobile,
            gmail: gmail,
        };

        try {
            if (studentToEdit) {
                const response = await axios.put(`${API_URL}${studentToEdit.roll_number}`, studentData);
                alert('Student updated successfully!');
                onStudentUpdated(response.data);
            } else {
                const response = await axios.post(API_URL, studentData);
                alert('Student added successfully!');
                onStudentAdded(response.data);
            }
            setName('');
            setDob('');
            setBranch('');
            setAddress('');
            setMobile('');
            setGmail('');
        } catch (err) {
            if (err.response && err.response.data && err.response.data.detail) {
                setError(err.response.data.detail);
            } else {
                setError('An unexpected error occurred. Please try again.');
            }
            console.error('Error submitting form:', err);
        }
    };

    return (
        <div className="form-container">
            <h2>{studentToEdit ? 'Edit Student' : 'Add New Student'}</h2>
            {error && <p className="error-message">{error}</p>}
            <form onSubmit={handleSubmit}>
                <div>
                    <label htmlFor="name">Name:</label>
                    <input
                        type="text"
                        id="name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label htmlFor="dob">Date of Birth:</label>
                    <input
                        type="date"
                        id="dob"
                        value={dob}
                        onChange={(e) => setDob(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label htmlFor="branch">Branch:</label>
                    <input
                        type="text"
                        id="branch"
                        value={branch}
                        onChange={(e) => setBranch(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label htmlFor="address">Address:</label>
                    <input
                        type="text"
                        id="address"
                        value={address}
                        onChange={(e) => setAddress(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label htmlFor="mobile">Mobile:</label>
                    <input
                        type="text"
                        id="mobile"
                        value={mobile}
                        onChange={(e) => setMobile(e.target.value)}
                        required
                    />
                </div>
                <div>
                    <label htmlFor="gmail">Gmail:</label>
                    <input
                        type="email"
                        id="gmail"
                        value={gmail}
                        onChange={(e) => setGmail(e.target.value)}
                        required
                    />
                </div>
                <button type="submit">{studentToEdit ? 'Update Student' : 'Add Student'}</button>
                {studentToEdit && (
                    <button type="button" onClick={() => onStudentUpdated(null)}>
                        Cancel Edit
                    </button>
                )}
            </form>
        </div>
    );
};

export default StudentForm;