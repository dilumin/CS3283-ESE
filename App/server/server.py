# Import necessary modules
from flask import Flask, Response, jsonify
import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from crack_detection import crack_detection
from flask_cors import CORS
import logging

# Flask app initialization
app = Flask(__name__)

# Disable logging of ultrakytics
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
# Apply CORS to this app
CORS(app)

# Global variable to control the video feed state
is_streaming = True

# YOLO model loading (update the model path if necessary)
model = YOLO("../../Model/train5/weights/best.pt")

# Directory for storing cropped crack images
output_folder = 'New_Last_Cracks'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Variables for frame counting and saving images
frame_count = 0
last_save_time = time.time()
save_interval = 3  # Save an image every 3 seconds

# Function to generate raw video frames from the camera
def generate_frames(camera):
    global is_streaming
    while is_streaming:
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame from camera")
            break
        if isinstance(frame, np.ndarray) and frame.size > 0:
            ret, buffer = cv2.imencode('.jpg', frame)
            if ret:
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Function to generate frames after YOLO crack detection
def generate_yolo_frames(input_frames):
    global frame_count, last_save_time

    for frame in input_frames:
        if isinstance(frame, np.ndarray) and frame.size > 0:
            results = model(frame)
            for result in results:
                for box in result.boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])

                    width = xmax - xmin
                    height = ymax - ymin

                    if width >= 300 or height >= 300:
                        current_time = time.time()

                        if current_time - last_save_time >= save_interval:
                            if ymin + 10 < ymax - 10 and xmin + 10 < xmax - 10:
                                cropped_crack = frame[ymin + 10:ymax - 10, xmin + 10:xmax - 10]

                                if cropped_crack.size == 0:
                                    print("Cropped crack image is empty, skipping...")
                                    continue

                                crack_image_name = f'crack_{frame_count}.png'
                                crack_image_path = os.path.join(output_folder, crack_image_name)

                                try:
                                    result_frame, perpendicular_distances = crack_detection(cropped_crack)
                                    print(f"Perpendicular distances: {perpendicular_distances}")
                                except Exception as e:
                                    print(f"Error in crack_detection: {e}")
                                    continue

                                if isinstance(result_frame, np.ndarray):
                                    ret, buffer = cv2.imencode('.jpg', result_frame)
                                    if ret:
                                        frame_for_stream = buffer.tobytes()
                                        yield (b'--frame\r\n'
                                               b'Content-Type: image/jpeg\r\n\r\n' + frame_for_stream + b'\r\n')

            time.sleep(0.5)  # Avoid high CPU usage

# Route to start the video feed
@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)
    global is_streaming
    is_streaming = True
    return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

# Home route
@app.get('/')
def index():
    return "Hello World!"

# Route for YOLO-based video processing
@app.route('/yolo')
def yolo():
    camera = cv2.VideoCapture(0)
    
    def generate_input_frames():
        while True:
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame from camera in input stream")
                break
            yield frame

    return Response(generate_yolo_frames(generate_input_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main driver function
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)