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

// Class names for the MobileNet-SSD model
const classNames = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"];

// --- 1. Main Function: Load the Model (UPDATED FOR ONNXRUNTIME-WEB) ---
async function main() {
    try {
        // Use ort.InferenceSession.create() instead of new onnx.Inference()
        // The global 'ort' object comes from the new script tag
        session = await ort.InferenceSession.create("./mobilenet_ssd.onnx");
        console.log("Model loaded successfully!");
        startBtn.disabled = false;
        startBtn.innerText = "Start Live Detection";
    } catch (error) {
        console.error("Failed to load the model:", error);
        alert("Error: Could not load the detection model. Check the console for details.");
    }
}

// --- 2. Event Listeners for Buttons and File Input ---
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
fileInput.addEventListener('change', detectFromFile);

// --- 3. Webcam and Live Detection Logic ---
async function startDetection() {
    // Check if the model loaded successfully before starting
    if (!session) {
        alert("Model is not loaded yet. Please wait or check the console for errors.");
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.style.display = 'block';
        canvas.style.display = 'none';
        isDetecting = true;
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
    const { boxes, scores, indices, personCount } = await runInference(video);
    drawDetections(video, boxes, scores, indices);

    if (personCount > 0) {
        resultBox.style.display = 'block';
        resultText.innerText = `Detection Complete - People: ${personCount}`;
        showAlert(personCount);
        console.log(`Detected ${personCount} people at ${new Date().toLocaleTimeString()}`);
    } else {
        resultBox.style.display = 'none';
    }

    animationFrameId = requestAnimationFrame(detect);
}

// --- 4. File-based Detection Logic ---
function detectFromFile(event) {
    if (!session) {
        alert("Model is not loaded yet. Please wait or check the console for errors.");
        return;
    }
    const file = event.target.files[0];
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    const image = new Image();
    image.src = imageUrl;

    image.onload = async () => {
        const { boxes, scores, indices, personCount } = await runInference(image);
        drawDetections(image, boxes, scores, indices);
        resultBox.style.display = 'block';
        resultText.innerText = `Detection Complete - People: ${personCount}`;
        showAlert(personCount);
        console.log(`Detected ${personCount} people in the uploaded image.`);
    };
}

// --- 5. Core Inference and Processing Function (UPDATED FOR ONNXRUNTIME-WEB) ---
async function runInference(source) {
    const inputTensor = preprocess(source);
    
    // The new API requires an object mapping input names to tensors
    // Replace "image" with the correct name from Netron
    const feeds = { "input": inputTensor }; // The input name 'image' might need to be adjusted based on the model
    
    // Run inference
    const results = await session.run(feeds);
    
    // The output tensor is in results.detection_out
    const outputData = results.detection_out.data;

    const { boxes, scores, indices, personCount } = postprocess(outputData, source.width, source.height);
    return { boxes, scores, indices, personCount };
}

function preprocess(source) {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    canvas.width = 300;
    canvas.height = 300;
    context.drawImage(source, 0, 0, 300, 300);
    const imageData = context.getImageData(0, 0, 300, 300);
    
    const data = new Float32Array(300 * 300 * 3);
    for (let i = 0; i < 300 * 300; i++) {
        data[i] = imageData.data[i * 4] / 255;
        data[i + 300 * 300] = imageData.data[i * 4 + 1] / 255;
        data[i + 300 * 300 * 2] = imageData.data[i * 4 + 2] / 255;
    }
    
    // Create the tensor using the new library's format
    return new ort.Tensor('float32', data, [1, 3, 300, 300]);
}

function postprocess(outputData, originalWidth, originalHeight) {
    const boxes = [];
    const scores = [];
    const indices = [];
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

            boxes.push([xmin, ymin, xmax - xmin, ymax - ymin]);
            scores.push(score);
            indices.push(labelIndex);
            personCount++;
        }
    }
    return { boxes, scores, indices, personCount };
}

// --- 6. Drawing Function ---
function drawDetections(source, boxes, scores, indices) {
    canvas.style.display = 'block';
    video.style.display = 'none';
    canvas.width = source.width || source.videoWidth;
    canvas.height = source.height || source.videoHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    
    for (let i = 0; i < boxes.length; i++) {
        const box = boxes[i];
        const score = scores[i];
        const label = classNames[indices[i]];

        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;
        ctx.strokeRect(box[0], box[1], box[2], box[3]);

        ctx.fillStyle = '#00FF00';
        ctx.font = '18px Orbitron';
        const text = `${label} (${Math.round(score * 100)}%)`;
        ctx.fillText(text, box[0], box[1] > 20 ? box[1] - 5 : 20);
    }
}

// --- Run the main function when the page loads ---
main();
