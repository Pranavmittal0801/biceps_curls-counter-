import os
from flask import Flask, request, Response, jsonify
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return "Pose detection server is running. Use /video_feed to see webcam stream or /upload_video to upload a video."

# ✅ Route to handle video uploads
@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    return jsonify({"message": "File uploaded successfully", "path": filepath}), 200

# ✅ Route to process uploaded video (Pose Detection)
@app.route('/process_video', methods=['POST'])
def process_video():
    data = request.json
    video_path = data.get('video_path')
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({"error": "Invalid video path"}), 400

    cap = cv2.VideoCapture(video_path)
    processed_frames = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # ✅ Convert frame to grayscale (example processing)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, buffer = cv2.imencode('.jpg', frame)
        processed_frames.append(buffer.tobytes())

    cap.release()
    
    return jsonify({"message": "Processing complete", "frames": len(processed_frames)}), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
