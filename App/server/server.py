# Importing necessary modules
from flask import Flask, Response, jsonify
import cv2
import os
import time
import numpy as np
from ultralytics import YOLO
from crack_detection import crack_detection
from flask_cors import CORS  # Import CORS

# Flask constructor takes the name of the current module (__name__) as an argument.
app = Flask(__name__)

# Apply CORS to this app
CORS(app)

# Capture video from the default camera (index 0)

# Global variable to control the video feed state
is_streaming = True

# Function to generate the video feed frame by frame
def generate_frames(camera):
    
    global is_streaming  # Reference the global variable
    while is_streaming:
        # Capture frame-by-frame
        success, frame = camera.read()
        if not success:
            print("Failed to capture frame from camera")
            break
        else:
            # Check if the frame is a valid numpy array
            if isinstance(frame, np.ndarray) and frame.size > 0:
                # Encode the frame as JPEG
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    # Return the frame as part of a multipart HTTP response
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                print("Captured frame is not a valid image.")

# Load your trained YOLOv8 segmentation model
model = YOLO("../../Model/train5/weights/best.pt")  # Update the path to your model if necessary

# Create a directory to store cropped crack images
output_folder = 'New_Last_Cracks'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_count = 0  # To uniquely name each saved cropped crack
last_save_time = time.time()  # Record the time of the last saved image
save_interval = 3  # Save a new photo every 1 second

def generate_yolo_frames(input_frames):
    global frame_count, last_save_time

    for frame in input_frames:
        # Ensure the frame is still a NumPy array before cropping or processing
        if isinstance(frame, np.ndarray) and frame.size > 0:
            # Perform crack detection with segmentation
            results = model(frame)

            # Extract detection results
            for result in results:
                for box in result.boxes:
                    xmin, ymin, xmax, ymax = map(int, box.xyxy[0])  # Bounding box coordinates

                    # Calculate width and height of the bounding box
                    width = xmax - xmin
                    height = ymax - ymin

                    # Check if both width and height are greater than or equal to 300
                    if width >= 300 or height >= 300:
                        current_time = time.time()

                        # Check if the save interval has passed since the last saved image
                        if current_time - last_save_time >= save_interval:
                            # Ensure that the coordinates are valid before cropping
                            if ymin + 10 < ymax - 10 and xmin + 10 < xmax - 10:
                                # Cropping should be done before encoding
                                cropped_crack = frame[ymin + 10:ymax - 10, xmin + 10:xmax - 10]

                                # Check if cropped image is valid
                                if cropped_crack.size == 0:
                                    print("Cropped crack image is empty, skipping...")
                                    continue  # Skip this iteration

                                # Generate a unique name for the crack image
                                crack_image_name = f'crack_{frame_count}.png'
                                crack_image_path = os.path.join(output_folder, crack_image_name)

                                # Call the crack_detection function
                                try:
                                    result_frame, perpendicular_distances = crack_detection(cropped_crack)
                                except Exception as e:
                                    print(f"Error in crack_detection: {e}")
                                    continue  # Skip processing on error

                                # Handle result_frame if it is valid
                                if isinstance(result_frame, np.ndarray):
                                    # Encode the processed frame as JPEG for streaming
                                    ret, buffer = cv2.imencode('.jpg', result_frame)
                                    if ret:
                                        # Convert to bytes for streaming
                                        frame_for_stream = buffer.tobytes()
                                        yield (b'--frame\r\n'
                                               b'Content-Type: image/jpeg\r\n\r\n' + frame_for_stream + b'\r\n')
                                else:
                                    print("Result frame is not valid, skipping...")

                    
# Route to start the video feed
@app.route('/video_feed')
def video_feed():
    camera = cv2.VideoCapture(0)
    global is_streaming
    is_streaming = True
    return Response(generate_frames(camera), mimetype='multipart/x-mixed-replace; boundary=frame')



@app.get('/')
def index():
    return "Hello World!"

@app.route('/yolo')
def yolo():
    camera = cv2.VideoCapture(0)
    
    def generate_input_frames():
        while True:
            
            success, frame = camera.read()
            if not success:
                print("Failed to capture frame from camera in input stream")
                break
            else:
                yield frame  # Yield each frame for processing in YOLO

    # Pass the frames to the YOLO function and stream the results
    return Response(generate_yolo_frames(generate_input_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

# Main driver function
if __name__ == '__main__':
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(host='0.0.0.0', port=5001, debug=True)