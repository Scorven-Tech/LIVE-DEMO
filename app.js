// Get references to our HTML elements
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');
const fileInput = document.getElementById('file-upload');
const video = document.getElementById('webcam');
const canvas = document.getElementById('output-canvas');
const resultText = document.getElementById('result-text');
const resultBox = document.getElementById('result-box');
const ctx = canvas.getContext('2d');

let session;
let isDetecting = false;
let animationFrameId;
let lastAnnouncedCount = -1;

// Class names for the MobileNet-SSD model
const classNames = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];

// --- 1. Main Function: Load the Model ---
async function main() {
    try {
        session = await ort.InferenceSession.create("./mobilenet_ssd.onnx");
        console.log("Model loaded successfully!");
        startBtn.disabled = false;
        startBtn.innerText = "Start Live Detection";
    } catch (error) {
        console.error("Failed to load the model:", error);
        alert("Error: Could not load the detection model. Check the console for details.");
    }
}

// --- 2. Event Listeners ---
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
fileInput.addEventListener('change', detectFromFile);

// --- 3. ALITA Voice Alert Function (Replaces pyttsx3) ---
function speak(text) {
    // Stop any previous speech to prevent overlap
    window.speechSynthesis.cancel();
    
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US'; // Set language
    utterance.rate = 1.0;     // Set speed
    
    // Use the browser's speech API
    window.speechSynthesis.speak(utterance);
}

// --- 4. Webcam and Live Detection Logic ---
async function startDetection() {
    if (!session) {
        alert("Model is not loaded yet. Please wait.");
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.style.display = 'block';
        canvas.style.display = 'none';
        isDetecting = true;
        lastAnnouncedCount = -1; // Reset counter on start
        detect();
    } catch (err) {
        console.error("Error accessing webcam:", err);
        alert("Could not access the webcam. Please grant permission.");
    }
}

function stopDetection() {
    isDetecting = false;
    cancelAnimationFrame(animationFrameId);
    if (video.srcObject) {
        video.srcObject.getTracks().forEach(track => track.stop());
        video.srcObject = null;
        video.style.display = 'none';
    }
    console.log("Detection stopped.");
}

async function detect() {
    if (!isDetecting || !session) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const { boxes, personCount } = await runInference(video);
    drawDetections(video, boxes, personCount);

    // Continue the loop
    animationFrameId = requestAnimationFrame(detect);
}

// --- 5. File-based Detection Logic ---
function detectFromFile(event) {
    if (!session) {
        alert("Model is not loaded yet. Please wait.");
        return;
    }
    const file = event.target.files[0];
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    const image = new Image();
    image.src = imageUrl;

    image.onload = async () => {
        const { boxes, personCount } = await runInference(image);
        drawDetections(image, boxes, personCount);
    };
}

// --- 6. Core Inference and Processing Function ---
async function runInference(source) {
    const inputTensor = preprocess(source);
    const feeds = { "input": inputTensor };
    const results = await session.run(feeds);

    // ✅ FIX: The output name is often just 'output' or 'detection_out'. 
    // We get the first output tensor from the results object directly.
    const outputTensor = results[Object.keys(results)[0]];
    const outputData = outputTensor.data;

    const { boxes, personCount } = postprocess(outputData, source.width, source.height);
    return { boxes, personCount };
}

function preprocess(source) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    const modelWidth = 224;
    const modelHeight = 224;
    
    canvas.width = modelWidth;
    canvas.height = modelHeight;
    context.drawImage(source, 0, 0, modelWidth, modelHeight);
    const imageData = context.getImageData(0, 0, modelWidth, modelHeight);
    
    const data = new Float32Array(modelWidth * modelHeight * 3);
    for (let i = 0; i < modelWidth * modelHeight; i++) {
        data[i] = imageData.data[i * 4] / 255;
        data[i + modelWidth * modelHeight] = imageData.data[i * 4 + 1] / 255;
        data[i + modelWidth * modelHeight * 2] = imageData.data[i * 4 + 2] / 255;
    }
    
    return new ort.Tensor('float32', data, [1, 3, modelHeight, modelWidth]);
}

function postprocess(outputData, originalWidth, originalHeight) {
    const boxes = [];
    let personCount = 0;
    
    for (let i = 0; i < outputData.length; i += 7) {
        const score = outputData[i + 2];
        if (score < 0.5) continue; 
        
        const labelIndex = outputData[i + 1];
        const label = classNames[labelIndex];

        if (label === "person") {
            const xmin = outputData[i + 3] * originalWidth;
            const ymin = outputData[i + 4] * originalHeight;
            const xmax = outputData[i + 5] * originalWidth;
            const ymax = outputData[i + 6] * originalHeight;
            
            const scoreText = `${Math.round(score * 100)}%`;
            boxes.push({ box: [xmin, ymin, xmax - xmin, ymax - ymin], label: 'person', score: scoreText });
            personCount++;
        }
    }
    return { boxes, personCount };
}

// --- 7. Drawing and Counting Function ---
function drawDetections(source, boxes, personCount) {
    canvas.style.display = 'block';
    video.style.display = 'none';
    canvas.width = source.width || source.videoWidth;
    canvas.height = source.height || source.videoHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    
    // Draw each box
    boxes.forEach(item => {
        const box = item.box;
        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;
        ctx.strokeRect(box[0], box[1], box[2], box[3]);

        ctx.fillStyle = '#00FF00';
        ctx.font = '18px Orbitron';
        const text = `${item.label} (${item.score})`;
        ctx.fillText(text, box[0], box[1] > 20 ? box[1] - 5 : 20);
    });

    // Update the count and call the alert functions
    if (personCount > 0) {
        resultBox.style.display = 'block';
        resultText.innerText = `Detection Complete - People: ${personCount}`;
        
        // This replaces the logic from your Flask app
        if (personCount !== lastAnnouncedCount) {
            console.log(`Detected ${personCount} people.`); // Replaces detect_log.txt
            showAlert(personCount); // From your alert.js
            speak(`${personCount} people detected`); // ALITA voice alert
            lastAnnouncedCount = personCount;
        }
    } else {
        resultBox.style.display = 'none';
        lastAnnouncedCount = 0;
    }
}

// --- Run the main function when the page loads ---
main();
