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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Global Variables for Sharing ---
latest_faces_data = {"verified": False, "faces": []}
camera_running = True
lock = asyncio.Lock()  # For safe access to shared data

# --- Data and Model Loading (Kept as before) ---
# Load known data
if os.path.exists("face_data.json"):
    with open("face_data.json", "r") as f:
        face_data = json.load(f)
else:
    face_data = []  # Start empty if file doesn't exist

known_encodings = [np.array(p["encoding"]) for p in face_data]
known_info = [(p["nome"], p["idade"], p["profissao"]) for p in face_data]

# Models
detector = dlib.get_frontal_face_detector()
# Check if model files exist
predictor_path = "shape_predictor_68_face_landmarks.dat"
face_rec_model_path = "dlib_face_recognition_resnet_model_v1.dat"

if not os.path.exists(predictor_path):
    logger.error(f"Error: Predictor file not found at {predictor_path}")
    logger.error("Make sure to download and place the .dat files in the backend/ directory.")
    exit()
if not os.path.exists(face_rec_model_path):
    logger.error(f"Error: Recognition model file not found at {face_rec_model_path}")
    logger.error("Make sure to download and place the .dat files in the backend/ directory.")
    exit()

predictor = dlib.shape_predictor(predictor_path)
face_rec_model = dlib.face_recognition_model_v1(face_rec_model_path)


# --- Face Processing Function (Kept as before) ---
def get_face_encodings(frame):
    encodings = []
    # Resize to speed up, but check if frame is not too small
    scale = 0.25
    height, width = frame.shape[:2]
    if width == 0 or height == 0:
        return [], frame  # Return empty if frame is invalid

    # Ensure frame doesn't have 0 dimension after resizing
    new_width = int(width * scale)
    new_height = int(height * scale)
    if new_width == 0 or new_height == 0:
        scale = 1.0  # Don't resize if it would be too small
        new_width = width
        new_height = height

    small_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    try:
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
         logger.error(f"Error converting color: {e}, shape: {small_frame.shape}")
         return [], frame  # Return empty if there's an error in conversion

    faces = detector(rgb_small)

    for face in faces:
        shape = predictor(rgb_small, face)
        encoding = face_rec_model.compute_face_descriptor(rgb_small, shape)
        # Pass the original face detected in the small frame and the encoding
        encodings.append((face, np.array(encoding), scale))  # Include scale
    return encodings, frame


# --- Main Camera Loop and Processing ---
async def run_camera_and_process():
    global latest_faces_data, camera_running

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Error: Could not open camera.")
        camera_running = False
        return

    logger.info("Camera opened. Press 'q' in the window to exit.")
    status_frame_counter = 0  # Renamed to avoid conflict

    while camera_running:
        ret, frame = cap.read()
        if not ret:
            logger.warning("Error: Could not receive frame. Trying again...")
            await asyncio.sleep(0.5)  # Wait a bit before trying again
            # Try to reopen the camera if it consistently fails? (Optional)
            # cap.release()
            # cap = cv2.VideoCapture(0)
            # if not cap.isOpened():
            #     logger.error("Error: Failed to reopen camera. Exiting.")
            #     camera_running = False
            #     break
            # continue
            # For simplicity, let's exit if reading fails repeatedly
            logger.error("Failed to read frame. Shutting down.")
            camera_running = False
            break


        encodings_data, processed_frame = get_face_encodings(frame.copy())  # Use copy to not draw on original yet
        current_faces = []
        status_color = (255, 255, 0)  # Yellow (analyzing)
        status_icon = "🔎"
        status_text = "ANALYZING"
        general_access_granted = False  # General scene status

        # Process each face found
        for face, encoding, scale_used in encodings_data:
            name = "Unknown"
            age = ""
            profession = ""
            match_found = False
            face_status_color = (0, 0, 255)  # Default red (blocked/unknown)

            if known_encodings:  # Only compare if there are known faces
                 matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=0.5)
                 if True in matches:
                     match_index = matches.index(True)
                     name, age, profession = known_info[match_index]
                     match_found = True
                     general_access_granted = True  # If a known face is found, grant access
                     face_status_color = (0, 255, 0)  # Green (granted)


            # Coordinates adjusted by scale
            inv_scale = 1.0 / scale_used
            x1 = int(face.left() * inv_scale)
            y1 = int(face.top() * inv_scale)
            x2 = int(face.right() * inv_scale)
            y2 = int(face.bottom() * inv_scale)

            # Draw border on original frame (processed_frame)
            thickness = 2 + ((status_frame_counter // 10) % 3)
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), face_status_color, thickness)

            # Draw info balloon
            texts = [f"Name: {name}"]
            if age: texts.append(f"Age: {age}")
            if profession: texts.append(f"Profession: {profession}")
            if not match_found: texts = ["Unknown"]

            max_text_width = 0
            total_text_height = 0
            text_sizes = []
            for t in texts:
                (w, h), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                max_text_width = max(max_text_width, w)
                total_text_height += h + 10  # Line spacing
                text_sizes.append(((w,h), _))

            balloon_width = max_text_width + 20
            balloon_height = total_text_height + 10  # Padding

            info_x_base = x1  # Align with left of face by default
            info_y = y1 - balloon_height - 10  # Position above face

            # Adjust if it goes off screen
            if info_y < 10: info_y = y2 + 10  # Place below if it doesn't fit above
            if info_x_base + balloon_width > processed_frame.shape[1] - 10:
                 info_x_base = processed_frame.shape[1] - balloon_width - 10  # Adjust to not go off right

            # Draw balloon background
            cv2.rectangle(processed_frame, (info_x_base, info_y), (info_x_base + balloon_width, info_y + balloon_height), (0, 0, 0), cv2.FILLED)

            # Draw text in balloon
            current_y = info_y + text_sizes[0][0][1] + 5  # Adjust initial Y for first text
            for i, text in enumerate(texts):
                 cv2.putText(processed_frame, text, (info_x_base + 10, current_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_status_color, 2)
                 if i < len(texts) - 1:
                      current_y += text_sizes[i+1][0][1] + 10  # Move to next line


            current_faces.append({
                "top": y1, "right": x2, "bottom": y2, "left": x1,
                "name": name, "age": age, "profession": profession
            })

        # Determine final general status
        if encodings_data:  # If any face was detected
             if general_access_granted:
                 status_color = (0, 255, 0)
                 status_icon = "✔"
                 status_text = "ACCESS GRANTED"
             else:
                 status_color = (0, 0, 255)
                 status_icon = "✖"
                 status_text = "ACCESS DENIED"
        # else: Keep "Analyzing" if no face was detected


        # Draw general status at the top
        if encodings_data or status_text == "ANALYZING":  # Show status even if no face is detected
            message = f"{status_icon}  {status_text}"
            (text_width, text_height), _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)  # Slightly smaller size
            box_height = text_height + 30
            box_coords = ((10, 10), (30 + text_width, 10 + box_height))
            # Draw with some transparency (alpha blending)
            overlay = processed_frame.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], status_color, thickness=cv2.FILLED)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, processed_frame, 1 - alpha, 0, processed_frame)
            # Draw text over rectangle
            cv2.putText(processed_frame, message, (20, 10 + text_height + 15), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)


        # Display processed frame
        cv2.imshow("Facial Recognition", processed_frame)
        status_frame_counter += 1

        # Update shared data for WebSocket
        async with lock:
            latest_faces_data = {
                "verified": len(encodings_data) > 0,
                "faces": current_faces,
                "status_text": status_text,  # Also send status
                "status_color": status_color  # And color (client can use)
            }

        # Check if 'q' was pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("Key 'q' pressed. Shutting down...")
            camera_running = False
            break  # Exit while loop

        await asyncio.sleep(0.01)  # Small pause to not overload CPU and allow other asyncio tasks

    # Release resources after exiting loop
    cap.release()
    cv2.destroyAllWindows()
    logger.info("Camera released and windows closed.")
    # Signal WebSocket to stop as well (if needed)
    # (Current logic will make the main server stop when this task finishes)


# --- WebSocket Handler (Only sends data) ---
async def handler(websocket, *args):
    logger.info(f"Client connected: {websocket.remote_address}")
    try:
        while camera_running:  # Continue sending while camera is active
            async with lock:
                data_to_send = latest_faces_data
            await websocket.send(json.dumps(data_to_send))
            await asyncio.sleep(0.1)  # Send updates every 100ms
    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client disconnected (OK): {websocket.remote_address}")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client disconnected (Error): {websocket.remote_address} - {e}")
    except Exception as e:
        logger.error(f"Error in WebSocket handler: {e}")
    finally:
        logger.info(f"Ending handler for {websocket.remote_address}")


# --- Main Async Function ---
async def main():
    # Start camera task in parallel
    camera_task = asyncio.create_task(run_camera_and_process())

    # Start WebSocket server
    # Use 0.0.0.0 to allow connections from other machines on local network (optional)
    server_address = "localhost"  #"0.0.0.0"
    server_port = 8765
    logger.info(f"Starting WebSocket server at ws://{server_address}:{server_port}")
    async with websockets.serve(handler, server_address, server_port):
         # Keep server running until camera task finishes
         await camera_task  # Wait for camera to complete (when 'q' is pressed)

    logger.info("Camera task completed. WebSocket server shutting down.")


# --- Entry Point ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down.")
    finally:
        # Ensure camera_running is False when exiting
        camera_running = False
        logger.info("Application terminated.")
