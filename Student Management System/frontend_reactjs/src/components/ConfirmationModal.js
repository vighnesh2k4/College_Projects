import React from 'react';

const ConfirmationModal = ({ message, onConfirm, onCancel }) => {
    return (
        <div className="modal">
            <div className="modal-content">
                <p>{message}</p>
                <button onClick={onConfirm}>Confirm</button>
                <button onClick={onCancel}>Cancel</button>
            </div>
        </div>
    );
};

export default ConfirmationModal;