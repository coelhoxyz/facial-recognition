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
from utils.logger import setup_logging

# --- Configure Logging ---
setup_logging()
logger = logging.getLogger(__name__)

# --- Global Variables for Sharing ---
latest_faces_data = {"verified": False, "faces": []}
camera_running = True
lock = asyncio.Lock()  # For safe access to shared data

# --- Data and Model Loading ---
# Load known face data
face_data_path = "src/data/face_data.json"
if os.path.exists(face_data_path):
    try:
        with open(face_data_path, "r") as f:
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

try:
    known_encodings = [np.array(p["encoding"]) for p in face_data]
    known_info = [(p["name"], p["age"], p["profession"]) for p in face_data]
    logger.info(f"Loaded {len(known_encodings)} known face encodings.")
except KeyError as e:
    logger.error(f"Missing key {e} in face_data.json. Ensure the JSON uses English keys ('name', 'age', 'profession').")
    known_encodings = []
    known_info = []
except Exception as e:
    logger.error(f"Error processing face data: {e}")
    known_encodings = []
    known_info = []

# dlib Models
detector = dlib.get_frontal_face_detector()

# --- Construct Paths Relative to Script Location --- 
script_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.dirname(script_dir) # Go one level up from src/

# Check if model files exist in the backend directory
predictor_filename = "src/shape_predictor_68_face_landmarks.dat"
face_rec_model_filename = "src/dlib_face_recognition_resnet_model_v1.dat"

predictor_path = os.path.join(backend_dir, predictor_filename)
face_rec_model_path = os.path.join(backend_dir, face_rec_model_filename)

# Update face_data.json path as well to be relative to backend_dir
face_data_path = os.path.join(backend_dir, "src/data/face_data.json")

# --- Load known face data ---
if os.path.exists(face_data_path):
    try:
        with open(face_data_path, "r") as f:
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

# Extract encodings and info (assuming keys 'encoding', 'name', 'age', 'profession' in JSON)
try:
    known_encodings = [np.array(p["encoding"]) for p in face_data]
    known_info = [(p["name"], p["age"], p["profession"]) for p in face_data]
    logger.info(f"Loaded {len(known_encodings)} known face encodings.")
except KeyError as e:
    logger.error(f"Missing key {e} in face_data.json. Ensure the JSON uses English keys ('name', 'age', 'profession').")
    known_encodings = []
    known_info = []
except Exception as e:
    logger.error(f"Error processing face data: {e}")
    known_encodings = []
    known_info = []

# dlib Models
detector = dlib.get_frontal_face_detector()
# Check if model files exist
if not os.path.exists(predictor_path):
    logger.error(f"Error: Predictor file not found at expected location: {predictor_path}")
    logger.error(f"Make sure {predictor_filename} is downloaded and placed in the '{os.path.basename(backend_dir)}/' directory.")
    exit()
if not os.path.exists(face_rec_model_path):
    logger.error(f"Error: Recognition model file not found at expected location: {face_rec_model_path}")
    logger.error(f"Make sure {face_rec_model_filename} is downloaded and placed in the '{os.path.basename(backend_dir)}/' directory.")
    exit()

try:
    predictor = dlib.shape_predictor(predictor_path)
    face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)
    logger.info("dlib models loaded successfully.")
except Exception as e:
    logger.critical(f"Failed to load dlib models: {e}")
    exit()


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
    if new_width == 0 or new_height == 0:
        logger.warning("Frame too small, cannot resize further. Using original size.")
        scale = 1.0
        new_width = width
        new_height = height

    small_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    try:
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
         logger.warning(f"Error converting color: {e}, shape: {small_frame.shape}")
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
            # Pass the dlib face object, the encoding, and the scale used
            encodings.append((face, np.array(encoding), scale))
        except Exception as e:
            logger.error(f"Error processing face landmarks or descriptor: {e}")
            continue # Skip this face

    return encodings, frame


# --- Main Camera Loop and Processing ---
async def run_camera_and_process():
    global latest_faces_data, camera_running

    camera_index = 0 # Or get from config
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error(f"Error: Could not open camera at index {camera_index}.")
        camera_running = False
        return

    logger.info(f"Camera {camera_index} opened. Press 'q' in the window to exit.")
    status_frame_counter = 0

    while camera_running:
        try:
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Error: Could not receive frame. Trying again...")
                await asyncio.sleep(0.5)
                # Simple retry; could add logic to re-initialize camera after N failures
                continue
        except Exception as e:
            logger.error(f"Error reading frame from camera: {e}")
            await asyncio.sleep(1) # Wait a bit longer after an exception
            continue

        try:
            encodings_data, processed_frame = get_face_encodings(frame.copy())
        except Exception as e:
            logger.error(f"Error in get_face_encodings: {e}")
            processed_frame = frame # Show the original frame if processing fails
            encodings_data = []

        current_faces_list = [] # Renamed from current_rostos
        status_color = (255, 255, 0)  # Yellow (analyzing)
        status_icon = "🔎"
        status_text = "ANALYZING"
        general_access_granted = False

        # Process each face found
        for face, encoding, scale_used in encodings_data:
            name = "Unknown"
            age = ""
            profession = ""
            match_found = False
            face_status_color = (0, 0, 255)  # Default red (blocked/unknown)

            if known_encodings:
                try:
                    matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                    face_distances = face_recognition.face_distance(known_encodings, encoding)
                    best_match_index = np.argmin(face_distances)

                    if matches[best_match_index]:
                        match_index = best_match_index # Use the best match index
                        name, age, profession = known_info[match_index]
                        match_found = True
                        general_access_granted = True
                        face_status_color = (0, 255, 0) # Green (granted)
                except Exception as e:
                    logger.warning(f"Error comparing faces: {e}")

            # Coordinates adjusted by scale
            inv_scale = 1.0 / scale_used
            x1 = int(face.left() * inv_scale)
            y1 = int(face.top() * inv_scale)
            x2 = int(face.right() * inv_scale)
            y2 = int(face.bottom() * inv_scale)

            # Clamp coordinates to frame boundaries
            h, w = processed_frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)

            # --- Draw bounding box --- 
            thickness = 2 + ((status_frame_counter // 10) % 3)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), face_status_color, thickness)

            # --- Draw info balloon --- 
            texts = [f"Name: {name}"]
            if age: texts.append(f"Age: {age}")
            if profession: texts.append(f"Profession: {profession}")
            if not match_found: texts = ["Unknown"]

            # Calculate text sizes and balloon dimensions
            max_text_width = 0
            total_text_height = 0
            line_height = 0
            text_sizes = []
            font_scale = 0.5
            font_thickness = 1
            font = cv2.FONT_HERSHEY_SIMPLEX
            line_spacing = 5 # Pixels between lines

            for t in texts:
                (w, h), baseline = cv2.getTextSize(t, font, font_scale, font_thickness)
                text_h = h + baseline # Total height for the line including baseline
                max_text_width = max(max_text_width, w)
                total_text_height += text_h + line_spacing
                text_sizes.append(((w, text_h), baseline))
                line_height = max(line_height, text_h) # Max height of a single line
            
            if text_sizes: # Remove last spacing
                total_text_height -= line_spacing

            padding = 10
            balloon_width = max_text_width + 2 * padding
            balloon_height = total_text_height + 2 * padding

            # Position balloon above face
            info_x_base = x1
            info_y = y1 - balloon_height - 5 # 5px margin above face

            # Adjust if it goes off screen
            if info_y < padding: info_y = y2 + 5 # Place below if not enough space above
            if info_x_base + balloon_width > w - padding:
                 info_x_base = w - balloon_width - padding # Adjust left if too wide
            if info_x_base < padding:
                 info_x_base = padding # Ensure it doesn't go off left
            info_y = max(padding, info_y) # Ensure it doesn't go off top

            # Draw balloon background (semi-transparent)
            try:
                sub_img = processed_frame[info_y:info_y + balloon_height, info_x_base:info_x_base + balloon_width]
                black_rect = np.zeros(sub_img.shape, dtype=np.uint8)
                res = cv2.addWeighted(sub_img, 0.5, black_rect, 0.5, 1.0)
                processed_frame[info_y:info_y + balloon_height, info_x_base:info_x_base + balloon_width] = res
            except Exception as e:
                # Fallback to solid if blending fails (e.g., near edge)
                cv2.rectangle(processed_frame, (info_x_base, info_y), (info_x_base + balloon_width, info_y + balloon_height), (0, 0, 0), cv2.FILLED)

            # Draw text in balloon
            current_y = info_y + padding + text_sizes[0][0][1] # Start Y for first line baseline
            for i, text in enumerate(texts):
                text_x = info_x_base + padding
                cv2.putText(processed_frame, text, (text_x, current_y), font, font_scale, face_status_color, font_thickness, cv2.LINE_AA)
                if i < len(texts) - 1:
                    current_y += text_sizes[i+1][0][1] + line_spacing # Move baseline for next line

            current_faces_list.append({
                "top": y1, "right": x2, "bottom": y2, "left": x1,
                "name": name, "age": age, "profession": profession
            })

        # Determine final general status
        if encodings_data:
             if general_access_granted:
                 status_color = (0, 255, 0) # Green
                 status_icon = "✔"
                 status_text = "ACCESS GRANTED"
             else:
                 status_color = (0, 0, 255) # Red
                 status_icon = "✖"
                 status_text = "ACCESS DENIED"
        # else: Keep "ANALYZING" (Yellow)

        # --- Draw general status banner at the top --- 
        if encodings_data or status_text == "ANALYZING":
            message = f"{status_icon}  {status_text}"
            font_scale_banner = 0.8
            font_thickness_banner = 2
            (text_width, text_height), baseline_banner = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, font_scale_banner, font_thickness_banner)
            banner_height = text_height + baseline_banner + 20 # 10px padding top/bottom
            banner_width = text_width + 40 # 20px padding left/right
            banner_x = 10
            banner_y = 10
            box_coords = ((banner_x, banner_y), (banner_x + banner_width, banner_y + banner_height))

            # Draw banner background (semi-transparent)
            try:
                sub_img_banner = processed_frame[banner_y:banner_y + banner_height, banner_x:banner_x + banner_width]
                color_rect_banner = np.full(sub_img_banner.shape, status_color, dtype=np.uint8)
                res_banner = cv2.addWeighted(sub_img_banner, 0.4, color_rect_banner, 0.6, 0)
                processed_frame[banner_y:banner_y + banner_height, banner_x:banner_x + banner_width] = res_banner
            except Exception as e:
                # Fallback to solid if blending fails
                 cv2.rectangle(processed_frame, box_coords[0], box_coords[1], status_color, thickness=cv2.FILLED)

            # Draw text over banner
            text_x_banner = banner_x + 20
            text_y_banner = banner_y + 10 + text_height # Position baseline
            cv2.putText(processed_frame, message, (text_x_banner, text_y_banner), cv2.FONT_HERSHEY_DUPLEX, font_scale_banner, (255, 255, 255), font_thickness_banner, cv2.LINE_AA)

        # Display processed frame
        cv2.imshow("Facial Recognition", processed_frame) # Changed window title
        status_frame_counter += 1

        # Update shared data for WebSocket
        async with lock:
            latest_faces_data = {
                "verified": len(encodings_data) > 0,
                "faces": current_faces_list,
                "status_text": status_text,
                "status_color": status_color
            }

        # Check if 'q' was pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("'q' key pressed. Shutting down...")
            camera_running = False
            break

        await asyncio.sleep(0.01)

    # Release resources after exiting loop
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Camera released and windows closed.")


# --- WebSocket Handler (Sends shared data) ---
async def handler(websocket, *args):
    peer_name = websocket.remote_address
    logger.info(f"Client connected: {peer_name}")
    try:
        while camera_running:
            async with lock:
                data_to_send = latest_faces_data
            # Add basic validation before sending
            if isinstance(data_to_send, dict) and "faces" in data_to_send:
                await websocket.send(json.dumps(data_to_send))
            else:
                logger.warning("Invalid data format in latest_faces_data, not sending.")
            await asyncio.sleep(0.1) # Send updates every 100ms
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client disconnected (OK): {peer_name}")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client disconnected (Error): {peer_name} - {e}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler for {peer_name}: {e}")
    finally:
        logger.info(f"Ending handler for {peer_name}")


# --- Main Async Function --- 
async def main():
    logger.info("Application starting...")
    # Start camera task in parallel
    camera_task = asyncio.create_task(run_camera_and_process())

    # Start WebSocket server
    server_address = "localhost"
    server_port = 8765
    logger.info(f"Starting WebSocket server at ws://{server_address}:{server_port}")
    try:
        async with websockets.serve(handler, server_address, server_port):
             await camera_task # Wait for camera task to complete (e.g., on 'q' press)
    except OSError as e:
        logger.critical(f"Failed to start WebSocket server on {server_address}:{server_port}: {e} (Address likely already in use)")
        # Try to stop the camera task gracefully if the server fails
        if camera_task and not camera_task.done():
            logger.info("Attempting to stop camera task...")
            global camera_running
            camera_running = False
            try:
                await asyncio.wait_for(camera_task, timeout=2.0) # Give it time to stop
            except asyncio.TimeoutError:
                logger.warning("Camera task did not stop gracefully within timeout.")
            except Exception as cam_e:
                 logger.error(f"Error while waiting for camera task to stop: {cam_e}")
        exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error starting WebSocket server: {e}")
        exit(1)

    logger.info("Camera task completed. WebSocket server shutting down.")


# --- Entry Point --- 
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down.")
    except Exception as e:
        logger.critical(f"Unhandled exception in main loop: {e}")
    finally:
        # Ensure camera_running is False when exiting
        camera_running = False # Set flag just in case
        logger.info("Application terminated.")
