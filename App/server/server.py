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
import requests

ESP32_CAMERA_URL = "http://172.20.10.3/capture"


# Flask app initialization
app = Flask(__name__)

# Disable logging of ultrakytics
logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
# Apply CORS to this app
CORS(app)


# Global variable to control the video feed state
is_streaming = True
perpendicular_distances_global = []

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
                                    perpendicular_distances_global.extend(perpendicular_distances)
                                except Exception as e:
                                    print(f"Error in crack_detection: {e}")
                                    continue

                                if isinstance(result_frame, np.ndarray):
                                    ret, buffer = cv2.imencode('.jpg', result_frame)
                                    if ret:
                                        frame_for_stream = buffer.tobytes()
                                        yield (b'--frame\r\n'
                                               b'Content-Type: image/jpeg\r\n\r\n' + frame_for_stream + b'\r\n')
                                        

            time.sleep(0.5)  
            

# Route to start the video feed
@app.route('/video_feed1')
def video_feed():
    camera = cv2.VideoCapture(0)
    global is_streaming
    is_streaming = True
    return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')

# Home route
@app.get('/')
def index():
    return "Hello World!"

def generate_frames_from_esp32_optimized():
    global is_streaming
    while is_streaming:
        try:
            response = requests.get(ESP32_CAMERA_URL, stream=True)
            byte_buffer = bytes()
            
            for chunk in response.iter_content(chunk_size=1024):
                byte_buffer += chunk
                
                # Detect end of the current frame
                if b'\xff\xd9' in byte_buffer:
                    # Find the index where frame ends
                    end_index = byte_buffer.index(b'\xff\xd9') + 2
                    frame_data = byte_buffer[:end_index]
                    byte_buffer = byte_buffer[end_index:]

                    # Convert the byte data to an OpenCV image
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                    if frame is not None:
                        ret, buffer = cv2.imencode('.jpg', frame)
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                    
        except Exception as e:
            print(f"Error fetching frames from ESP32 camera: {e}")
            break



@app.route('/video_feed')
def optimized_capture():
    global is_streaming
    is_streaming = True
    return Response(generate_frames_from_esp32_optimized(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Route for YOLO-based video processing
@app.route('/yolo1')
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

def new_yolo_see(generating_img):
    global frame_count, last_save_time
    for frame in generating_img:
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
                                    perpendicular_distances_global.extend(perpendicular_distances)
                                except Exception as e:
                                    print(f"Error in crack_detection: {e}")
                                    continue

                                if isinstance(result_frame, np.ndarray):
                                    ret, buffer = cv2.imencode('.jpg', result_frame)
                                    if ret:
                                        frame_for_stream = buffer.tobytes()
                                        yield (b'--frame\r\n'
                                               b'Content-Type: image/jpeg\r\n\r\n' + frame_for_stream + b'\r\n')
                                        

            time.sleep(0.5)

# Route for YOLO-based video processing coming from esp32
@app.route('/yolo')
def yolo1():
    global is_streaming
    is_streaming = True
    def generating_img():
        try:
            while is_streaming:
                response = requests.get(ESP32_CAMERA_URL, stream=True)
                byte_buffer = bytes()

                for chunk in response.iter_content(chunk_size=1024):
                    byte_buffer += chunk

                    if b'\xff\xd9' in byte_buffer:
                        end_index = byte_buffer.index(b'\xff\xd9') + 2
                        frame_data = byte_buffer[:end_index]
                        byte_buffer = byte_buffer[end_index:]

                        frame = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                        yield frame
        except Exception as e:
            print(f"Error in YOLO detection from ESP32: {e}")
    return Response(new_yolo_see(generating_img()), mimetype='multipart/x-mixed-replace; boundary=frame')
            





@app.get('/data')
def data():
    global perpendicular_distances_global
    temp = perpendicular_distances_global
    perpendicular_distances_global = []
    return jsonify(temp)

# Route to stop the video feed

# Route to stop the video feed
@app.get('/reset')
def reset_data():
    global perpendicular_distances_global
    perpendicular_distances_global = []
    return "Reset successful"


# Main driver function
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)