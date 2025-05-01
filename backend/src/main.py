import asyncio
import websockets
import cv2
import face_recognition
import os
import numpy as np
import json
import logging
from utils.logger import setup_logging

# --- Setup Logging ---
setup_logging()
logger = logging.getLogger(__name__)

def load_faces():
    faces = []
    names = []
    folder = 'src/data/face_data' # Assumes this folder is relative to where the script is run
    if not os.path.exists(folder):
        logger.error(f"Folder '{folder}' not found.")
        return faces, names

    logger.info(f"Loading faces from folder: {folder}")
    loaded_count = 0
    skipped_count = 0
    for filename in os.listdir(folder):
        if filename.startswith('.'): # Skip hidden files like .DS_Store
            continue
        path = os.path.join(folder, filename)
        if not os.path.isfile(path):
            continue # Skip subdirectories

        try:
            image = face_recognition.load_image_file(path)
            locations = face_recognition.face_locations(image) # Use default model
            # Use the first face found in the image
            if locations:
                encodings = face_recognition.face_encodings(image, [locations[0]]) # Encode only the first face
                if encodings:
                    faces.append(encodings[0])
                    # Use filename (without extension) as the name
                    names.append(os.path.splitext(filename)[0])
                    loaded_count += 1
                else:
                    logger.warning(f"Could not generate encoding for the face found in {filename}")
                    skipped_count += 1
            else:
                logger.warning(f"No face detected in {filename}")
                skipped_count += 1
        except Exception as e:
            logger.error(f"Error processing file {filename}: {e}")
            skipped_count += 1

    logger.info(f"Finished loading faces. Loaded: {loaded_count}, Skipped/Errors: {skipped_count}")
    return faces, names

known_faces, known_names = load_faces()

# Initialize camera
try:
    camera_index = 0
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise IOError(f"Cannot open camera at index {camera_index}")
    logger.info(f"Camera {camera_index} initialized successfully.")
except Exception as e:
    logger.critical(f"Failed to initialize camera: {e}")
    exit()


async def recognize(websocket, path): # Added path argument (unused but required by websockets.serve)
    peer_name = websocket.remote_address
    logger.info(f"Client connected: {peer_name}")
    try:
        while True:
            ret, frame = camera.read()
            if not ret or frame is None:
                logger.warning("Could not read frame from camera.")
                await asyncio.sleep(0.1) # Avoid busy-waiting on error
                continue

            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert BGR (OpenCV) to RGB (face_recognition)
            rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            results = []
            verified = False
            face_detected = False

            try:
                locations = face_recognition.face_locations(rgb, model="cnn") # Using CNN model
                face_detected = bool(locations)

                if not locations:
                    # Send empty results if no faces detected
                    await websocket.send(json.dumps({"faces": [], "verified": False}))
                    await asyncio.sleep(0.05) # Short delay
                    continue

                encodings = face_recognition.face_encodings(rgb, locations)

                for (top, right, bottom, left), encoding in zip(locations, encodings):
                    matches = face_recognition.compare_faces(known_faces, encoding, tolerance=0.6) # Default tolerance
                    name = "Unknown"

                    # Use face_distance to find the best match
                    face_distances = face_recognition.face_distance(known_faces, encoding)
                    if len(face_distances) > 0:
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_names[best_match_index]
                            verified = True # Set to True if any known face is recognized

                    # Scale back coordinates to original frame size
                    results.append({
                        "top": top * 4,
                        "right": right * 4,
                        "bottom": bottom * 4,
                        "left": left * 4,
                        "name": name
                    })

            except Exception as e:
                logger.error(f"Error during face recognition process: {e}")
                # Send empty result on error to avoid crashing client?
                await websocket.send(json.dumps({"faces": [], "verified": False}))
                await asyncio.sleep(0.1)
                continue

            # Send results via WebSocket
            await websocket.send(json.dumps({
                "faces": results,
                "verified": verified
            }))
            await asyncio.sleep(0.05) # Control frame rate / CPU usage

    except websockets.exceptions.ConnectionClosedOK:
        logger.info(f"Client disconnected (OK): {peer_name}")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.warning(f"Client disconnected (Error): {peer_name} - {e}")
    except Exception as e:
        logger.error(f"Unexpected error in recognize handler for {peer_name}: {e}")
    finally:
        logger.info(f"Ending recognize handler for {peer_name}")

async def main():
    host = "localhost"
    port = 8765
    try:
        async with websockets.serve(recognize, host, port):
            logger.info(f"WebSocket server started at ws://{host}:{port}")
            await asyncio.Future()  # Run forever
    except OSError as e:
         logger.critical(f"Failed to start WebSocket server on {host}:{port}: {e} (Address likely in use)")
    except Exception as e:
         logger.critical(f"Unexpected error starting WebSocket server: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
         logger.critical(f"Unhandled exception in main execution: {e}")
    finally:
        if 'camera' in locals() and camera.isOpened():
             camera.release()
             logger.info("Camera released.")
        logger.info("Application terminated.")
