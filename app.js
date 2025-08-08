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

// --- 1. Main Function: Load the Model ---
async function main() {
    try {
        // Create an ONNX inference session
        session = new onnx.Inference({ backendHint: 'webgl' });
        // Load the model. Make sure the path is correct.
        await session.loadModel("./mobilenet_ssd.onnx");
        console.log("Model loaded successfully!");
        startBtn.disabled = false;
        startBtn.innerText = "Start Live Detection";
    } catch (error) {
        console.error("Failed to load the model:", error);
        alert("Error: Could not load the detection model.");
    }
}

// --- 2. Event Listeners for Buttons and File Input ---
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);
fileInput.addEventListener('change', detectFromFile);


// --- 3. Webcam and Live Detection Logic ---
async function startDetection() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        video.style.display = 'block'; // Show the video feed
        canvas.style.display = 'none'; // Hide the canvas for now
        isDetecting = true;
        detect(); // Start the detection loop
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
    if (!isDetecting) return;

    // Set canvas dimensions to match the video
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Pre-process the frame and run inference
    const { boxes, scores, indices, personCount } = await runInference(video);

    // Draw the results
    drawDetections(video, boxes, scores, indices);

    // Update the count and call your alert function
    if (personCount > 0) {
        resultBox.style.display = 'block';
        resultText.innerText = `Detection Complete - People: ${personCount}`;
        
        // This is where we integrate your files!
        showAlert(personCount); // From alert.js
        console.log(`Detected ${personCount} people at ${new Date().toLocaleTimeString()}`); // Alternative to detect_log.txt
    } else {
        resultBox.style.display = 'none';
    }


    // Continue the loop
    animationFrameId = requestAnimationFrame(detect);
}

// --- 4. File-based Detection Logic ---
function detectFromFile(event) {
    const file = event.target.files[0];
    if (!file) return;

    const imageUrl = URL.createObjectURL(file);
    const image = new Image();
    image.src = imageUrl;

    image.onload = async () => {
        // Run inference on the static image
        const { boxes, scores, indices, personCount } = await runInference(image);
        
        // Draw the results
        drawDetections(image, boxes, scores, indices);

        // Update the count and call your alert function
        resultBox.style.display = 'block';
        resultText.innerText = `Detection Complete - People: ${personCount}`;
        
        // Integration!
        showAlert(personCount); // From alert.js
        console.log(`Detected ${personCount} people in the uploaded image.`); // Alternative to detect_log.txt
    };
}


// --- 5. Core Inference and Processing Function ---
async function runInference(source) {
    // Pre-process the image data
    const inputTensor = preprocess(source);

    // Run inference
    const outputMap = await session.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;

    // Post-process the output
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
    
    // Convert image data to a tensor
    const data = new Float32Array(300 * 300 * 3);
    for (let i = 0; i < 300 * 300; i++) {
        data[i] = imageData.data[i * 4] / 255;
        data[i + 300 * 300] = imageData.data[i * 4 + 1] / 255;
        data[i + 300 * 300 * 2] = imageData.data[i * 4 + 2] / 255;
    }
    
    return new onnx.Tensor(data, 'float32', [1, 3, 300, 300]);
}

function postprocess(outputData, originalWidth, originalHeight) {
    const boxes = [];
    const scores = [];
    const indices = [];
    let personCount = 0;
    
    // The MobileNet-SSD output is a flat array [image_id, label, score, xmin, ymin, xmax, ymax]
    for (let i = 0; i < outputData.length; i += 7) {
        const score = outputData[i + 2];
        if (score < 0.5) continue; // Confidence threshold

        const labelIndex = outputData[i + 1];
        const label = classNames[labelIndex];

        // We only care about detecting "person"
        if (label === "person") {
            const xmin = outputData[i + 3] * originalWidth;
            const ymin = outputData[i + 4] * originalHeight;
            const xmax = outputData[i + 5] * originalWidth;
            const ymax = outputData[i + 6] * originalHeight;

            boxes.push([xmin, ymin, xmax - xmin, ymax - ymin]); // [x, y, width, height]
            scores.push(score);
            indices.push(labelIndex);
            personCount++;
        }
    }
    return { boxes, scores, indices, personCount };
}


// --- 6. Drawing Function ---
function drawDetections(source, boxes, scores, indices) {
    // Show canvas and set its size
    canvas.style.display = 'block';
    video.style.display = 'none'; // Hide video if it's running
    canvas.width = source.width || source.videoWidth;
    canvas.height = source.height || source.videoHeight;
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
    
    // Draw each box
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
