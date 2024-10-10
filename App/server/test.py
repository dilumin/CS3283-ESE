
# Function to generate frames from ESP32 after YOLO crack detection
def generate_yolo_frames_from_esp32():
    global frame_count, last_save_time

    byte_buffer = bytes()

    while True:
        try:
            # Fetch frames from ESP32
            response = requests.get(ESP32_CAMERA_URL, stream=True)
            for chunk in response.iter_content(chunk_size=1024):
                byte_buffer += chunk

                # Detect the end of the current frame
                if b'\xff\xd9' in byte_buffer:
                    end_index = byte_buffer.index(b'\xff\xd9') + 2
                    frame_data = byte_buffer[:end_index]
                    byte_buffer = byte_buffer[end_index:]

                    # Convert the byte data to an OpenCV image
                    frame = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        
                        # Perform YOLO detection on the frame
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
                                                ret, buffer = cv2.imencode('.jpg', frame)
                                                if ret:
                                                    frame_for_stream = buffer.tobytes()
                                                    yield (b'--frame\r\n'
                                                           b'Content-Type: image/jpeg\r\n\r\n' + frame_for_stream + b'\r\n')

                    # Sleep for a bit to simulate processing time
                    time.sleep(0.5)

        except Exception as e:
            print(f"Error in YOLO detection from ESP32: {e}")
            break




def generate_frames_yolo_esp2():
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

                                                # crack_image_name = f'crack_{frame_count}.png'
                                                # crack_image_path = os.path.join(output_folder, crack_image_name)

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
        except:
            print(f"Error fetching frames from ESP32 camera: {e}")
            break
                                                        

# New route for YOLO detection from ESP32 camera
@app.route('/yolo')
def yolo_esp32():
    global is_streaming
    is_streaming = True
    return Response(generate_frames_yolo_esp2(), mimetype='multipart/x-mixed-replace; boundary=frame')



