from flask import Flask, render_template, request, Response, redirect, url_for, send_from_directory
import cv2
import os
import numpy as np
from werkzeug.utils import secure_filename
import pyttsx3
import threading
import time
from queue import Queue

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
LOG_FILE = 'detection_log.txt'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load MobileNetSSD model
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt", "MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
           "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
THRESHOLD = 10

camera = None
live_detection = False
latest_count = 0

# Text-to-speech queue
speech_queue = Queue()

def speaker_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            print("[Speech] Speaking:", text)
            engine = pyttsx3.init(driverName='sapi5')
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print("[Speech Error]", e)
        speech_queue.task_done()

speaker_thread = threading.Thread(target=speaker_worker, daemon=False)
speaker_thread.start()

def repeat_speaker():
    def speak_loop():
        global latest_count
        last_announced = -1
        while True:
            time.sleep(5)
            if live_detection and latest_count > 0 and latest_count != last_announced:
                print(f"[Repeater] {latest_count} people detected")
                speech_queue.put(f"{latest_count} people detected")
                last_announced = latest_count
    thread = threading.Thread(target=speak_loop, daemon=True)
    thread.start()

repeat_speaker()

def detect_people(frame):
    global latest_count
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    count = 0
    print("---- Detection Results ----")
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        idx = int(detections[0, 0, i, 1])
        detected_class = CLASSES[idx] if idx < len(CLASSES) else "unknown"
        print(f"Detection {i}: Class={detected_class}, Confidence={confidence:.2f}")
        if confidence > 0.5 and detected_class == "person":
            count += 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    latest_count = count
    print("[Detection] People detected:", count)

    try:
        with open(LOG_FILE, "w") as f:
            f.write(f"People detected: {count}\n")
    except Exception as e:
        print("[Log Write Error]", e)

    cv2.putText(frame, f"People: {count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    if count >= THRESHOLD:
        cv2.putText(frame, "\u26a0 Crowd Alert!", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return frame, count

def generate_frames():
    global camera, live_detection
    while live_detection:
        success, frame = camera.read()
        if not success:
            break
        frame, _ = detect_people(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start')
def start():
    global live_detection, camera
    if not live_detection:
        camera = cv2.VideoCapture(0)
        live_detection = True
    return redirect(url_for('index'))

@app.route('/stop')
def stop():
    global live_detection, camera
    live_detection = False
    if camera is not None:
        camera.release()
        camera = None
    return redirect(url_for('index'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    frame = cv2.imread(filepath)
    if frame is not None:
        frame, count = detect_people(frame)
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        cv2.imwrite(output_path, frame)
        return render_template('index.html', upload_result=filename, count=count)
    return redirect(url_for('index'))

@app.route('/outputs/<filename>')
def send_output(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)

if __name__ == '__main__':
    try:
        app.run(debug=True, host='0.0.0.0')
    finally:
        speech_queue.put(None)
        speaker_thread.join()