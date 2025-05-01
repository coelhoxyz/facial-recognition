import cv2
import face_recognition
import json
import os
import logging
from utils.logger import setup_logging

JSON_FILE = "src/data/face_data.json"

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

def load_database():
    if os.path.exists(JSON_FILE):
        try:
            with open(JSON_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {JSON_FILE}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading {JSON_FILE}: {e}")
            return []
    return []

def save_database(database):
    try:
        with open(JSON_FILE, "w") as f:
            json.dump(database, f, indent=2)
        logger.info(f"Database saved successfully to {JSON_FILE}")
    except Exception as e:
        logger.error(f"Error saving database to {JSON_FILE}: {e}")

def capture_face():
    video = cv2.VideoCapture(0)
    logger.info("Press 'c' to capture the image or 'q' to quit.")

    while True:
        ret, frame = video.read()
        if not ret:
            continue

        cv2.imshow("Face Capture", frame)
        key = cv2.waitKey(1)

        if key & 0xFF == ord('c'):
            video.release()
            cv2.destroyAllWindows()
            return frame
        elif key & 0xFF == ord('q'):
            video.release()
            cv2.destroyAllWindows()
            exit()

def process_face(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_recognition.face_locations(rgb)

    if len(faces) != 1:
        logger.error("No face or multiple faces detected. Please try again.")
        return None

    encoding = face_recognition.face_encodings(rgb, known_face_locations=faces)[0]
    return encoding.tolist()

def main():
    logger.info("=== Facial Registration ===")
    name = input("Full name: ")
    age = input("Age: ")
    occupation = input("Occupation: ")

    frame = capture_face()
    encoding = process_face(frame)

    if encoding is None:
        return

    new_person = {
        "name": name,  # Keeping field names consistent with existing code
        "age": age,
        "profession": occupation,
        "encoding": encoding
    }

    database = load_database()
    database.append(new_person)
    save_database(database)

    logger.info(f"[SUCCESS] {name}'s face saved successfully!")

if __name__ == "__main__":
    main()
