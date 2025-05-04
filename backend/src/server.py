import cv2
import dlib
import face_recognition
import numpy as np
import asyncio
import websockets
import json
import os
import time
import logging
from .utils.logger import setup_logging
import threading
from flask import Flask, Response, render_template_string, jsonify, request
import argparse
import concurrent.futures

# --- Configure Logging ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Flask App Setup ---
flask_app = Flask(__name__)
output_frame = None # Global variable to hold the latest frame for streaming
# Use the same lock for thread-safe access to output_frame
lock = asyncio.Lock() # Reuse the existing lock (or create a separate threading.Lock if preferred)

# --- Global Variables for Sharing ---
latest_faces_data = {"verified": False, "faces": []}
camera_running = False # Start as False, set True when camera starts
selected_camera_idx = None # Store the chosen camera index
camera_task_handle = None # To hold the asyncio task for the camera loop
latest_raw_frame = None # Add global for the latest raw frame before overlays

# --- Data and Model Loading ---
# Ensure paths are correct (relative to backend/)
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir)
predictor_filename = "src/shape_predictor_68_face_landmarks.dat"
face_rec_model_filename = "src/dlib_face_recognition_resnet_model_v1.dat"
predictor_path = os.path.join(backend_dir, predictor_filename)
face_rec_model_path = os.path.join(backend_dir, face_rec_model_filename)
face_data_path = os.path.join(script_dir, "data", "face_data.json") # Expect data directly in backend/

# dlib Models
detector = dlib.get_frontal_face_detector()
try:
    # Verify paths before loading
    if not os.path.exists(predictor_path): raise FileNotFoundError(f"Predictor not found at {predictor_path}")
    if not os.path.exists(face_rec_model_path): raise FileNotFoundError(f"Recognition model not found at {face_rec_model_path}")
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
    logger.info("dlib models loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load dlib models: {e}")
    exit()

# Load known face data
if os.path.exists(face_data_path):
    try:
        with open(face_data_path, "r", encoding='utf-8') as f:
            face_data = json.load(f)
        logger.info(f"Known face data loaded from {face_data_path}")
    except json.JSONDecodeError:
        logger.error(f"Error decoding {face_data_path}. The file might be corrupted.")
        face_data = []
    except Exception as e:
        logger.error(f"Unexpected error loading {face_data_path}: {e}")
        face_data = []
else:
    logger.warning(f"File {face_data_path} not found. No known faces loaded.")
    face_data = []

# Extract encodings and info
try:
    known_encodings = [np.array(p["encoding"]) for p in face_data]
    known_info = [(p["name"], p["age"], p["profession"]) for p in face_data] # Expects English keys now
    logger.info(f"Loaded {len(known_encodings)} known face encodings.")
except KeyError as e:
    logger.error(f"Missing key '{e}' in face_data.json. Ensure the JSON uses English keys ('name', 'age', 'profession', 'encoding').")
    known_encodings = []
    known_info = []
except Exception as e:
    logger.error(f"Error processing face data: {e}")
    known_encodings = []
    known_info = []

# --- Function to Load Face Data (Also updates globals) ---
def load_face_data():
    global known_encodings, known_info, face_data_path
    logger.info(f"Loading face data from: {face_data_path}")
    encodings = []
    info = []
    if os.path.exists(face_data_path):
        try:
            with open(face_data_path, "r", encoding='utf-8') as f:
                face_data = json.load(f)
            if not isinstance(face_data, list):
                logger.error(f"Data in {face_data_path} is not a list. Ignoring file.")
                face_data = []
            
            # Extract data
            for p in face_data:
                try:
                    encodings.append(np.array(p["encoding"]))
                    info.append((p["name"], p["age"], p["profession"])) # Assumes English keys
                except KeyError as e:
                    logger.warning(f"Skipping record due to missing key '{e}' in {face_data_path}: {p.get('name', 'N/A')}")
                except Exception as inner_e:
                    logger.warning(f"Skipping record due to error processing encoding/info in {face_data_path}: {inner_e}")

            logger.info(f"Successfully loaded {len(encodings)} known faces.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding {face_data_path}. File might be corrupted.")
        except Exception as e:
            logger.error(f"Unexpected error loading {face_data_path}: {e}")
    else:
        logger.warning(f"File {face_data_path} not found. No known faces loaded.")
    
    # Update global variables (needs lock? Reading mostly, but initial load)
    # Let's assume initial load is fine, but updates need lock.
    known_encodings = encodings
    known_info = info

# --- Function to Save Face Data --- 
def save_face_data(all_face_data):
    global face_data_path
    logger.info(f"Attempting to save {len(all_face_data)} records to {face_data_path}")
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(face_data_path), exist_ok=True)
        with open(face_data_path, "w", encoding='utf-8') as f:
            json.dump(all_face_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Face data saved successfully.")
        return True
    except Exception as e:
        logger.error(f"Error saving face data to {face_data_path}: {e}")
        return False

# --- Function to Process Single Face for Registration ---
def process_face_for_registration(frame):
    """Detects a single face and returns encoding. Logs errors."""
    if frame is None:
        logger.error("process_face_for_registration received None frame.")
        return None
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Use HOG for potentially faster registration detection?
        # Or stick to CNN? Let's use HOG here for speed.
        faces = face_recognition.face_locations(rgb, model="hog") 

        if len(faces) == 0:
            logger.error("Registration failed: No face detected.")
            return None # Indicate no face found
        elif len(faces) > 1:
            logger.error("Registration failed: Multiple faces detected.")
            return None # Indicate multiple faces

        encoding = face_recognition.face_encodings(rgb, known_face_locations=[faces[0]])[0]
        logger.info("Face encoding generated successfully for registration.")
        return encoding.tolist()
    except Exception as e:
        logger.error(f"Error during face processing for registration: {e}")
        return None # Indicate general error

# --- Face Processing Function ---
def get_face_encodings(frame):
    encodings = []
    scale = 0.25 # Resize scale
    height, width = frame.shape[:2]
    if width == 0 or height == 0:
        logger.warning("Received an invalid frame (width or height is 0).")
        return [], frame # Return empty if frame is invalid

    # Ensure frame doesn't have 0 dimension after resizing
    new_width = int(width * scale)
    new_height = int(height * scale)
    if new_width <= 0 or new_height <= 0: # Check for non-positive dimensions
        logger.warning(f"Frame too small ({width}x{height}), cannot resize further. Using original size.")
        scale = 1.0
        new_width = width
        new_height = height
        if width <= 0 or height <=0: # Cannot proceed if original is invalid
             return [], frame

    try:
        small_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
         logger.warning(f"Error during frame resize/convert: {e}, shape: {frame.shape}, target:({new_width},{new_height})")
         return [], frame

    try:
        faces = detector(rgb_small)
    except Exception as e:
        logger.error(f"Error during face detection: {e}")
        return [], frame

    for face in faces:
        try:
            shape = predictor(rgb_small, face)
            encoding = face_rec_model.compute_face_descriptor(rgb_small, shape)
            encodings.append((face, np.array(encoding), scale))
        except Exception as e:
            logger.error(f"Error processing face landmarks or descriptor: {e}")
            continue # Skip this face

    return encodings, frame


# --- Function to Find Available Cameras ---
def find_available_cameras(max_cameras_to_check=5):
    """Tries to open cameras by index to see which ones are available."""
    available_cameras = []
    logger.info(f"Probing for available cameras (checking indices 0 to {max_cameras_to_check-1})...")
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap is not None and cap.isOpened():
            logger.debug(f"Camera index {i} seems available.")
            available_cameras.append(i)
            cap.release()
        else:
             logger.debug(f"Camera index {i} not available or failed to open.")
             if cap is not None: cap.release()
    logger.info(f"Found available camera indices: {available_cameras}")
    return available_cameras


# --- Main Camera Loop and Processing ---
async def run_camera_and_process(camera_index):
    global latest_faces_data, camera_running, output_frame, latest_raw_frame, lock

    # Use the camera index passed as an argument
    cap = cv2.VideoCapture(camera_index) 
    if not cap.isOpened():
        logger.error(f"Error: Could not open camera at index {camera_index}.")
        camera_running = False
        return

    logger.info(f"Camera {camera_index} opened. Streaming video feed.")
    status_frame_counter = 0

    while camera_running:
        frame_start_time = time.monotonic()
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Error: Could not receive frame. Trying again...")
                await asyncio.sleep(0.5)
                continue
        except Exception as e:
            logger.error(f"Error reading frame from camera: {e}")
            await asyncio.sleep(1)
            continue

        raw_frame_copy = frame.copy() # Keep a copy before drawing overlays

        # --- Face processing and overlay drawing ---
        try:
            encodings_data, processed_frame = get_face_encodings(raw_frame_copy)
        except Exception as e:
            logger.error(f"Error in get_face_encodings: {e}")
            processed_frame = raw_frame_copy # Use original frame if processing fails
            encodings_data = []

        # --- Get frame dimensions *before* drawing loops ---
        h, w = processed_frame.shape[:2]

        current_faces_list = []
        status_color = (255, 255, 0)  # Yellow
        status_icon = "🔎"
        status_text = "ANALYZING"
        general_access_granted = False

        for face, encoding, scale_used in encodings_data:
            name = "Unknown"
            age = ""
            profession = ""
            match_found = False
            face_status_color = (0, 0, 255) # Red

            if known_encodings:
                try:
                    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_encodings, encoding)
                    if len(face_distances) > 0: # Check if there are known encodings to compare against
                        best_match_index = np.argmin(face_distances)
                        # Check if the best match distance is within tolerance
                        if matches[best_match_index] and face_distances[best_match_index] <= 0.5:
                            match_index = best_match_index
                            name, age, profession = known_info[match_index]
                            match_found = True
                            general_access_granted = True
                            face_status_color = (0, 255, 0) # Green
                except IndexError:
                     logger.error("IndexError during face comparison - mismatch between known_info and known_encodings?")
                except Exception as e:
                    logger.warning(f"Error comparing faces: {e}")

            # --- Drawing calculations ---
            inv_scale = 1.0 / scale_used
            x1 = int(face.left() * inv_scale)
            y1 = int(face.top() * inv_scale)
            x2 = int(face.right() * inv_scale)
            y2 = int(face.bottom() * inv_scale)
            x1 = max(0, x1); y1 = max(0, y1); x2 = min(w - 1, x2); y2 = min(h - 1, y2)

            # --- Draw Bounding Box ---
            thickness = 2 + ((status_frame_counter // 10) % 3)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), face_status_color, thickness)

            # --- Draw Info Balloon ---
            texts = [f"Name: {name}"]
            if age: texts.append(f"Age: {age}")
            if profession: texts.append(f"Profession: {profession}")
            if not match_found: texts = ["Unknown"]

            max_text_width = 0
            total_text_height = 0
            line_height = 0
            text_sizes = []
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_spacing = 5
            padding = 5 # Reduced padding

            for t in texts:
                (tw, th), baseline = cv2.getTextSize(t, font, font_scale, font_thickness)
                text_h = th + baseline
                max_text_width = max(max_text_width, tw)
                total_text_height += text_h + line_spacing
                text_sizes.append(((tw, text_h), baseline))
                line_height = max(line_height, text_h)
            if text_sizes: total_text_height -= line_spacing

            balloon_width = max_text_width + 2 * padding
            balloon_height = total_text_height + 2 * padding
            info_x_base = x1
            info_y = y1 - balloon_height - 2 # Reduced margin

            if info_y < padding: info_y = y2 + 2
            if info_x_base + balloon_width > w - padding: info_x_base = w - balloon_width - padding
            if info_x_base < padding: info_x_base = padding
            info_y = max(padding, info_y)

            # Ensure balloon coordinates are valid before drawing
            if info_y + balloon_height <= h and info_x_base + balloon_width <= w:
                try: # Draw semi-transparent background
                    sub_img = processed_frame[info_y:info_y + balloon_height, info_x_base:info_x_base + balloon_width]
                    black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                    res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
                    processed_frame[info_y:info_y + balloon_height, info_x_base:info_x_base + balloon_width] = res
                except Exception as draw_err: # Fallback to solid if blending fails
                     logger.debug(f"Balloon background blending failed: {draw_err}")
                     cv2.rectangle(processed_frame, (info_x_base, info_y), (info_x_base + balloon_width, info_y + balloon_height), (0, 0, 0), cv2.FILLED)

                current_y = info_y + padding + text_sizes[0][0][1] # Baseline of first text
                for i, text in enumerate(texts):
                    text_x = info_x_base + padding
                    cv2.putText(processed_frame, text, (text_x, current_y), font, font_scale, face_status_color, font_thickness, cv2.LINE_AA)
                    if i < len(texts) - 1: current_y += text_sizes[i+1][0][1] + line_spacing
            else:
                 logger.debug(f"Skipping balloon draw due to invalid coordinates: xy({info_x_base},{info_y}) wh({balloon_width},{balloon_height}) frame:({w}x{h})")

            current_faces_list.append({
                "top": y1, "right": x2, "bottom": y2, "left": x1,
                "name": name, "age": age, "profession": profession
            })

        # --- Determine and Draw General Status Banner ---
        if encodings_data:
             if general_access_granted:
                 status_color = (0, 255, 0); status_icon = "✔"; status_text = "ACCESS GRANTED"
             else:
                 status_color = (0, 0, 255); status_icon = "✖"; status_text = "ACCESS DENIED"

        if encodings_data or status_text == "ANALYZING":
            message = f"{status_icon}  {status_text}"
            font_scale_banner = 0.7 # Slightly smaller
            font_thickness_banner = 1
            (text_width, text_height), baseline_banner = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, font_scale_banner, font_thickness_banner)
            banner_padding = 10
            banner_height = text_height + baseline_banner + banner_padding
            banner_width = text_width + 2 * banner_padding
            banner_x = 10
            banner_y = 10
            box_coords = ((banner_x, banner_y), (banner_x + banner_width, banner_y + banner_height))

            # Ensure banner coordinates are valid
            if banner_y + banner_height <= h and banner_x + banner_width <= w:
                try: # Draw semi-transparent banner
                    sub_img_banner = processed_frame[banner_y:banner_y + banner_height, banner_x:banner_x + banner_width]
                    color_rect_banner = np.full(sub_img_banner.shape, status_color, dtype=np.uint8)
                    res_banner = cv2.addWeighted(sub_img_banner, 0.4, color_rect_banner, 0.6, 0)
                    processed_frame[banner_y:banner_y + banner_height, banner_x:banner_x + banner_width] = res_banner
                except Exception as banner_draw_err: # Fallback to solid
                     logger.debug(f"Banner background blending failed: {banner_draw_err}")
                     cv2.rectangle(processed_frame, box_coords[0], box_coords[1], status_color, thickness=cv2.FILLED)

                text_x_banner = banner_x + banner_padding
                text_y_banner = banner_y + int(banner_padding/2) + text_height # Position baseline within padding
                cv2.putText(processed_frame, message, (text_x_banner, text_y_banner), cv2.FONT_HERSHEY_DUPLEX, font_scale_banner, (255, 255, 255), font_thickness_banner, cv2.LINE_AA)
            else:
                logger.debug("Skipping banner draw due to invalid coordinates.")

        # --- End of drawing ---
        status_frame_counter += 1

        # --- Update shared data for WebSocket and MJPEG stream ---
        # Using try-finally to ensure lock is always released
        await lock.acquire()
        try:
            latest_raw_frame = raw_frame_copy # Store the raw frame
            latest_faces_data = {
                "verified": len(encodings_data) > 0,
                "faces": current_faces_list,
                "status_text": status_text,
                "status_color": status_color
            }
            # Encode the processed frame to JPEG and update the global variable
            (flag, encoded_image) = cv2.imencode(".jpg", processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80]) # Adjust quality
            if flag:
                output_frame = encoded_image.tobytes()
            else:
                logger.warning("Could not encode frame to JPEG.")
        finally:
            lock.release()

        # --- Control loop speed ---
        elapsed_time = time.monotonic() - frame_start_time
        desired_delay = 0.03 # ~33fps target
        actual_delay = max(0, desired_delay - elapsed_time)
        await asyncio.sleep(actual_delay) # Adjust sleep time based on processing time


    # Release resources after exiting loop
    cap.release()
    logger.info("Camera processing loop stopped.")


# --- WebSocket Handler ---
async def handler(websocket, *args):
    peer_name = websocket.remote_address
    logger.info(f"Client connected via WebSocket: {peer_name}")
    try:
        while camera_running:
            data_to_send = None
            await lock.acquire()
            try:
                data_to_send = latest_faces_data
            finally:
                lock.release()

            if isinstance(data_to_send, dict) and "faces" in data_to_send:
                try:
                    await websocket.send(json.dumps(data_to_send))
                except websockets.exceptions.ConnectionClosed:
                    # Break inner loop if connection closed during send
                    break
            else:
                logger.warning("Invalid data format in latest_faces_data, not sending via WebSocket.")
            await asyncio.sleep(0.1) # Send WebSocket updates less frequently
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"WebSocket client disconnected (OK): {peer_name}")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"WebSocket client disconnected (Error): {peer_name} - {e}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for {peer_name}: {e}")
    finally:
        logger.info(f"Ending WebSocket handler for {peer_name}")

# --- Flask Routes ---

# Template needs updating for camera selection UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Facial Recognition</title>
    <style>
        /* Combined Styling */
        body { font-family: sans-serif; margin: 1em; background-color: #282c34; color: #abb2bf; display: flex; flex-direction: column; align-items: center;}
        h1, h2 { text-align: center; color: #61afef; margin-bottom: 0.5em;}
        button { padding: 8px 15px; margin: 5px; cursor: pointer; border-radius: 4px; border: 1px solid #61afef; background-color: #3a3f4b; color: #abb2bf; }
        button:hover { background-color: #4b5260; }
        input[type=text], input[type=number] { padding: 8px; margin: 5px 0; border-radius: 4px; border: 1px solid #444; background-color: #3a3f4b; color: #abb2bf; width: calc(100% - 18px); }
        .container { background: #333842; padding: 1em; border-radius: 5px; margin-bottom: 1em; width: 90%; max-width: 700px; }
        #camera-selection { text-align: center; }
        #video-container { display: none; position: relative; width: fit-content; margin: 1em auto; }
        #video-stream { display: block; border: 1px solid #444; max-width: 100%; height: auto; background-color: #000;}
        #status-overlay { display: none; position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.6); color: white; padding: 5px 10px; border-radius: 4px; font-size: 0.9em; }
        #registration-form { display: none; }
        #registration-form label { display: block; margin-top: 10px; font-size: 0.9em; }
        #ws-info { display: none; margin-top: 1em; background:#333842; padding: 0.5em 1em; border:1px solid #444; border-radius: 4px; font-size: 0.8em;}
        #ws-info h2 { margin-top: 0; margin-bottom: 0.5em; font-size: 1em; color: #98c379; border-bottom: 1px solid #444; padding-bottom: 0.3em;}
        #ws-data { max-height: 150px; overflow-y: auto; white-space: pre-wrap; word-break: break-all; }
        .status-message { margin-top: 10px; padding: 10px; border-radius: 4px; text-align: center; font-weight: bold; }
        .success { background-color: #2a5c2a; color: lightgreen; }
        .error { background-color: #6e3030; color: salmon; }
    </style>
</head>
<body>
    <h1>Live Facial Recognition</h1>

    <div id="camera-selection" class="container">
        <h2>Select Camera</h2>
        <div id="camera-list">Loading cameras...</div>
        <p id="selection-error" class="error" style="display: none;"></p>
    </div>

    <div id="video-container">
        <img id="video-stream" src="#" width="640" height="480" alt="Loading video stream...">
        <div id="status-overlay">Status: Connecting...</div>
    </div>
    
    <div id="registration-form" class="container">
         <h2>Register New Face</h2>
         <p>Position face clearly in the video feed above and fill details.</p>
         <div>
             <label for="reg-name">Name:</label>
             <input type="text" id="reg-name" name="name" required>
         </div>
         <div>
             <label for="reg-age">Age:</label>
             <input type="number" id="reg-age" name="age">
         </div>
         <div>
             <label for="reg-profession">Profession:</label>
             <input type="text" id="reg-profession" name="profession">
         </div>
         <button id="register-button">Register This Face</button>
         <div id="registration-status"></div>
    </div>

    <div id="ws-info" class="container">
        <h2>WebSocket Data</h2>
        <pre id="ws-data">Not connected.</pre>
    </div>

    <script>
        // IDs for elements
        const cameraListDiv = document.getElementById('camera-list');
        const videoContainer = document.getElementById('video-container');
        const videoStream = document.getElementById('video-stream');
        const wsInfoDiv = document.getElementById('ws-info');
        const statusOverlay = document.getElementById('status-overlay');
        const selectionError = document.getElementById('selection-error');
        const regForm = document.getElementById('registration-form');
        const regName = document.getElementById('reg-name');
        const regAge = document.getElementById('reg-age');
        const regProfession = document.getElementById('reg-profession');
        const regButton = document.getElementById('register-button');
        const regStatus = document.getElementById('registration-status');
        const wsStatus = statusOverlay;
        const wsData = document.getElementById('ws-data');
        
        let socket;
        let selectedCameraIndex = null;

        // Fetch available cameras
        async function fetchCameras() {
            try {
                const response = await fetch('/api/cameras');
                if (!response.ok) {
                    throw new Error(`HTTP error! Status: ${response.status}`);
                }
                const cameras = await response.json();
                console.log("Available cameras:", cameras);
                cameraListDiv.innerHTML = ''; // Clear loading message
                if (cameras.length === 0) {
                    cameraListDiv.textContent = 'No cameras found.';
                    return;
                }
                cameras.forEach(index => {
                    const button = document.createElement('button');
                    button.textContent = `Camera ${index}`;
                    button.onclick = () => selectCamera(index);
                    cameraListDiv.appendChild(button);
                });
            } catch (error) {
                console.error('Error fetching cameras:', error);
                cameraListDiv.textContent = 'Error loading camera list.';
            }
        }

        // Select camera and start stream/backend process
        async function selectCamera(index) {
            if (selectedCameraIndex !== null) return;
            console.log(`Camera ${index} selected.`);
            selectedCameraIndex = index;
            selectionError.style.display = 'none';
            cameraListDiv.innerHTML = `Starting camera ${index}...`;

            try {
                const response = await fetch('/api/select_camera', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ camera_index: index })
                });
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(`Failed to start. Status: ${response.status}. ${errorData.error || ''}`);
                }
                console.log("Camera start confirmed by backend.");

                // Show relevant sections
                document.getElementById('camera-selection').style.display = 'none';
                videoContainer.style.display = 'block';
                regForm.style.display = 'block';
                wsInfoDiv.style.display = 'block';
                statusOverlay.style.display = 'block';
                videoStream.src = `{{ url_for('video_feed') }}?t=${new Date().getTime()}`;
                connectWebSocket();

            } catch (error) {
                console.error('Error selecting camera:', error);
                selectionError.textContent = `Error: ${error.message}`;
                selectionError.style.display = 'block';
                cameraListDiv.innerHTML = '';
                fetchCameras(); // Refresh list
                selectedCameraIndex = null;
            }
        }
        
        // Handle Registration Button Click
        regButton.onclick = async () => {
            const name = regName.value.trim();
            const age = regAge.value.trim();
            const profession = regProfession.value.trim();

            if (!name) {
                setRegStatus("Name is required.", true);
                return;
            }

            setRegStatus("Processing registration...", false);
            regButton.disabled = true;

            try {
                 const response = await fetch('/api/register', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: name, age: age, profession: profession })
                 });
                 const result = await response.json(); // Always try to parse JSON
                 
                 if (!response.ok) {
                    throw new Error(result.error || `Registration failed with status ${response.status}`);
                 }
                 
                 console.log("Registration response:", result);
                 setRegStatus(result.message || "Registration successful!", false);
                 // Clear form on success
                 regName.value = '';
                 regAge.value = '';
                 regProfession.value = '';

            } catch (error) {
                 console.error("Registration error:", error);
                 setRegStatus(`Error: ${error.message}`, true);
            } finally {
                 regButton.disabled = false; // Re-enable button
            }
        };
        
        // Helper to display registration status
        function setRegStatus(message, isError) {
             regStatus.textContent = message;
             regStatus.className = isError ? 'status-message error' : 'status-message success';
             // Auto-clear message after a few seconds?
             setTimeout(() => { regStatus.textContent = ''; regStatus.className='status-message'; }, isError ? 8000 : 5000);
        }

        // WebSocket connection logic (Can remain mostly the same)
        function connectWebSocket() { /* ... same as before ... */ }
        // --- WebSocket Handlers --- 
        // ... (Copy onopen, onmessage, onerror, onclose from previous version) ...

        window.onload = fetchCameras;
        
        // Video stream error handler (Can remain the same)
        videoStream.onerror = () => { /* ... same as before ... */ };

    </script>
</body>
</html>
"""

@flask_app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def generate_mjpeg():
    """Generates the MJPEG stream from the global output_frame."""
    global output_frame
    logger.info("MJPEG stream client connected.")
    while camera_running: # Stream only while camera is intended to run
        frame_to_send = None
        # --- Simple non-blocking read of the latest frame ---
        # This avoids locking issues between async loop and sync thread
        # but might occasionally miss a frame or show a slightly older one.
        frame_to_send = output_frame
        # ---

        if frame_to_send:
            try:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_to_send + b'\r\n')
            except Exception as stream_err:
                 logger.warning(f"Error sending frame to MJPEG client: {stream_err}")
                 # Client likely disconnected, break loop
                 break
        # Control streaming rate - sleep even if no new frame to avoid busy-waiting
        time.sleep(0.03) # ~33fps target

    logger.info("MJPEG stream client disconnected.")


@flask_app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_mjpeg(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- NEW Flask API Routes ---
@flask_app.route('/api/cameras')
def get_available_cameras():
    """Returns a list of available camera indices."""
    try:
        cameras = find_available_cameras()
        return jsonify(cameras)
    except Exception as e:
        logger.error(f"Error finding cameras: {e}")
        return jsonify({"error": "Failed to detect cameras"}), 500

@flask_app.route('/api/select_camera', methods=['POST'])
def select_camera_api():
    """Receives the selected camera index and starts the processing task."""
    global selected_camera_idx, camera_running, camera_task_handle
    
    data = request.get_json()
    if not data or 'camera_index' not in data:
        return jsonify({"error": "Missing 'camera_index' in request"}), 400

    try:
        chosen_index = int(data['camera_index'])
        logger.info(f"Received request to start camera index: {chosen_index}")

        # Basic validation (optional: re-check if it's in available_cams?)
        if chosen_index < 0:
            raise ValueError("Camera index cannot be negative.")

        # --- Critical Section: Check/Start Camera Task ---
        async def start_task_if_needed():
            nonlocal chosen_index # Access outer scope variable
            global selected_camera_idx, camera_running, camera_task_handle
            async with lock:
                if camera_task_handle and not camera_task_handle.done():
                    logger.warning(f"Camera task already running ({selected_camera_idx}). Ignoring request for {chosen_index}.")
                    return False, "Camera task already running."
                
                logger.info(f"Starting camera processing task for index {chosen_index}.")
                selected_camera_idx = chosen_index
                camera_running = True
                loop = main_event_loop
                if loop is None or not loop.is_running():
                    logger.error("Main event loop is not available or not running!")
                    raise RuntimeError("Cannot schedule task: Event loop not running.")
                
                # Create and store the task handle
                camera_task_handle = loop.create_task(run_camera_and_process(selected_camera_idx))
                return True, "Camera task started."
        
        # Run the async check/start logic within the current event loop
        # Since this is a sync Flask handler, we need run_coroutine_threadsafe
        loop = main_event_loop
        if loop is None or not loop.is_running():
            logger.error("Main event loop is not available or not running!")
            raise RuntimeError("Cannot schedule task: Event loop not running.")
        
        future = asyncio.run_coroutine_threadsafe(start_task_if_needed(), loop)
        started, message = future.result(timeout=5.0) # Wait for the async code to finish (increased timeout)
        # -- End Critical Section ---

        if started:
             return jsonify({"message": message}), 200
        else:
             # Return a conflict status if already running
             return jsonify({"error": message}), 409 

    except ValueError:
        logger.error(f"Invalid camera index received: {data.get('camera_index')}")
        return jsonify({"error": "Invalid camera index format"}), 400
    except Exception as e:
        logger.exception(f"Error starting camera task for index {data.get('camera_index')}")
        return jsonify({"error": "Failed to start camera processing"}), 500

# --- NEW Registration API Endpoint ---
@flask_app.route('/api/register', methods=['POST'])
def register_face_api():
    """Handles new face registration requests."""
    global latest_raw_frame, lock # Need lock here too
    
    data = request.get_json()
    if not data or not data.get('name'):
        return jsonify({"error": "Missing required field: name"}), 400
        
    name = data['name']
    age = data.get('age', '') # Optional fields
    profession = data.get('profession', '')
    
    logger.info(f"Registration request received for: {name}")
    
    current_raw_frame = None
    encoding_list = None
    message = ""
    success = False
    loop = asyncio.get_running_loop() # Get the running loop

    # --- Get frame (needs access to loop) ---
    async def get_latest_frame():
        nonlocal current_raw_frame # Modify outer variable
        async with lock:
            if latest_raw_frame is not None:
                 current_raw_frame = latest_raw_frame.copy()
            else:
                 current_raw_frame = None

    try:
        # Schedule getting the frame on the loop and wait for it
        frame_future = asyncio.run_coroutine_threadsafe(get_latest_frame(), loop)
        frame_future.result(timeout=1.0) # Wait max 1 sec for the frame
        
        if current_raw_frame is None:
             raise ValueError("Could not capture current frame from camera feed. Try again.")

        logger.info("Processing captured frame for registration...")
        encoding_list = process_face_for_registration(current_raw_frame)
        
        if encoding_list is None:
            raise ValueError("Failed to detect a single clear face or generate encoding.")
            
        logger.info(f"Encoding generated for {name}. Proceeding to save.")
        # --- End frame processing ---
        
        # --- Data Update Section (Sync file I/O, Async global update) --- 
        new_person = {
            "name": name,
            "age": age,
            "profession": profession,
            "encoding": encoding_list
        }
        
        # --- Perform synchronous file operations first --- 
        current_database = []
        if os.path.exists(face_data_path):
             try:
                 with open(face_data_path, "r", encoding='utf-8') as f:
                     current_database = json.load(f)
                 if not isinstance(current_database, list):
                     logger.error(f"Invalid data format in {face_data_path}, resetting.")
                     current_database = []
             except Exception as load_err:
                  logger.error(f"Error loading data before save: {load_err}")
                  current_database = [] 
        # --- End file load ---

        if any(p.get("name") == name for p in current_database):
            logger.warning(f"Name '{name}' already exists. Appending as a new entry.")
            
        current_database.append(new_person)
        
        if not save_face_data(current_database):
            raise IOError("Failed to save the updated face data to file.")
        # --- End file save ---
            
        # --- Now schedule the ASYNC update of global in-memory lists --- 
        async def update_globals_async():
            global known_encodings, known_info
            async with lock: 
                 known_encodings.append(np.array(encoding_list))
                 known_info.append((name, age, profession))
                 logger.info(f"Async: In-memory face data updated for {name}.")
                 
        global_update_future = asyncio.run_coroutine_threadsafe(update_globals_async(), loop)
        global_update_future.result(timeout=1.0) # Wait for global update
        logger.info(f"Finished updating globals for {name}.")
        # --- End global update ---

        message = f"User '{name}' registered successfully."
        success = True
            
    except ValueError as ve:
         message = str(ve)
         logger.error(f"Registration ValueError for {name}: {message}")
    except IOError as ioe:
         message = str(ioe)
         logger.error(f"Registration IOError for {name}: {message}")
    except concurrent.futures.TimeoutError:
         message = "Operation timed out (getting frame or updating globals)."
         logger.error(f"Registration timeout for {name}.")
    except Exception as e:
         message = "An unexpected error occurred during registration."
         logger.exception(f"Unexpected registration error for {name}") # Log traceback
         
    if success:
        return jsonify({"message": message}), 201 # 201 Created
    else:
        return jsonify({"error": message}), 500 # Internal Server Error or specific error

# --- Function to Run Flask in Thread ---
def run_flask_app(stop_event, loop):
    """Runs the Flask app in a separate thread and signals on failure."""
    flask_port = 5001
    logger.info(f"Starting Flask server for video stream on http://0.0.0.0:{flask_port}")
    try:
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR) # Keep Flask logs quiet
        flask_app.run(host='0.0.0.0', port=flask_port, debug=False, threaded=True, use_reloader=False)
    except Exception as e:
        logger.critical(f"Flask server failed: {e}")
        # Signal the main asyncio loop that Flask failed
        loop.call_soon_threadsafe(stop_event.set)
    finally:
        logger.info("Flask server thread stopped.")
        # If Flask stops for any reason (even non-error shutdown), signal main loop.
        # This might cause a shutdown if Flask is stopped externally.
        if not stop_event.is_set(): 
            loop.call_soon_threadsafe(stop_event.set)

# --- Main Async Function (Modified) ---
async def main():
    logger.info("Application starting... Web UI at http://localhost:5001")
    load_face_data() # Load data initially
    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    # Start Flask app (provides API endpoints and stream)
    flask_thread = threading.Thread(target=run_flask_app, args=(stop_event, loop), daemon=True)
    flask_thread.start()

    # Don't start camera_task here anymore
    # camera_task = asyncio.create_task(run_camera_and_process(selected_camera_index))

    server_address = "0.0.0.0"
    server_port = 8765
    logger.info(f"Starting WebSocket server at ws://{server_address}:{server_port}")
    websocket_server = None

    global camera_task_handle # Need access to potentially await it

    try:
        async with websockets.serve(handler, server_address, server_port) as server:
            websocket_server = server
            logger.info("Servers started. Waiting for camera selection or shutdown signal.")
            
            # Wait primarily for the stop_event now, as camera starts later
            await stop_event.wait()
            logger.info("Stop event received, initiating shutdown...")

    except OSError as e:
        logger.critical(f"Failed to start WebSocket server on {server_address}:{server_port}: {e}")
    except Exception as e:
        logger.critical(f"Unexpected error in main async setup: {e}")
    finally:
        logger.info("Main function finally block starting cleanup...")
        global camera_running
        if camera_running:
            logger.debug("Setting camera_running to False in main finally block.")
            camera_running = False # Ensure camera loop stops

        if websocket_server:
            logger.info("Closing WebSocket server...")
            websocket_server.close()
            try:
                 await asyncio.wait_for(websocket_server.wait_closed(), timeout=2.0)
                 logger.info("WebSocket server closed.")
            except asyncio.TimeoutError:
                 logger.warning("WebSocket server did not close within timeout.")

        # Wait for camera task to finish if it was started
        if camera_task_handle and not camera_task_handle.done():
            logger.info("Waiting for camera task to complete shutdown...")
            try:
                await asyncio.wait_for(camera_task_handle, timeout=2.0)
                logger.info("Camera task completed during shutdown.")
            except asyncio.TimeoutError:
                 logger.warning("Camera task did not finish within timeout during shutdown.")
                 camera_task_handle.cancel()
            except asyncio.CancelledError:
                 logger.info("Camera task was cancelled during shutdown.")
            except Exception as cam_ex:
                 logger.error(f"Exception while waiting for camera task during shutdown: {cam_ex}")

        logger.info("Main function cleanup finished.")


# --- Global reference to the main event loop ---
main_event_loop = None

# --- Entry Point --- (Modified)
if __name__ == "__main__":
    # Remove terminal camera selection logic
    # logger.info(f"Proceeding with camera index: {selected_index}")

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    main_event_loop = loop # Store the loop globally
    main_task = None

    def handle_exception(loop, context):
        msg = context.get("exception", context["message"])
        logger.critical(f"Caught unexpected exception in event loop: {msg}", exc_info=context.get('exception'))
    loop.set_exception_handler(handle_exception)

    try:
        logger.debug("Creating main task.")
        # main no longer takes camera index argument here
        main_task = loop.create_task(main())
        logger.debug("Running event loop forever.")
        loop.run_forever()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Main finally block: Initiating final shutdown.")
        if main_task and not main_task.done():
            logger.info("Cancelling main task...")
            main_task.cancel()
            try:
                loop.run_until_complete(main_task)
                logger.info("Main task completed after cancellation.")
            except asyncio.CancelledError:
                logger.info("Main task explicitly cancelled.")
            except Exception as final_ex:
                logger.error(f"Exception during final main task await: {final_ex}")

        logger.info("Stopping event loop.")
        if loop.is_running():
            loop.stop()
        logger.info("Application terminated.")
