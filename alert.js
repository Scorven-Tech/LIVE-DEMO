function showAlert(count) {
    if (count >= 10) {
        const alertBox = document.createElement("div");
        alertBox.className = "popup-alert";
        alertBox.innerText = "⚠️ Crowd Alert! Possible Stampede Detected (" + count + ")";

        document.body.appendChild(alertBox);

        setTimeout(() => {
            alertBox.remove();
        }, 5000);
    }
}

// Optional: Sound alert
function playAlertSound() {
    const audio = new Audio('/static/alert.mp3');
    audio.play();
}

// Style for the alert
const style = document.createElement('style');
style.innerHTML = `
.popup-alert {
    position: fixed;
    top: 20px;
    right: 20px;
    background: rgba(255, 0, 0, 0.9);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 15px 25px;
    border-radius: 10px;
    box-shadow: 0 0 20px red;
    z-index: 10000;
    animation: pulse 1s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.1); }
    100% { transform: scale(1); }
}
`;
document.head.appendChild(style);
