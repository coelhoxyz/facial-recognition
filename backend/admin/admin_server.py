from flask import Flask, request, jsonify, send_from_directory
import json
import os
import logging
from ..src.utils.logger import setup_logging
from flask_cors import CORS


app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Caminho para o arquivo JSON (agora relativo à raiz do backend)
JSON_FILE = os.path.join(os.path.dirname(__file__), '..', 'face_data.json')

@app.route('/')
def index():
    logger.info("Serving index.html")
    return send_from_directory('.', 'index.html')

@app.route('/styles.css')
def styles():
    logger.debug("Serving styles.css")
    return send_from_directory('.', 'styles.css')

def load_users():
    """Loads user data from the JSON file."""
    try:
        if os.path.exists(JSON_FILE):
            with open(JSON_FILE, 'r', encoding='utf-8') as f:
                users = json.load(f)
                logger.info(f"Loaded {len(users)} users from {JSON_FILE}")
                if not isinstance(users, list):
                    logger.error(f"Data in {JSON_FILE} is not a list. Returning empty list.")
                    return []
                return users
        else:
            logger.warning(f"File {JSON_FILE} not found, returning empty list.")
            return []
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {JSON_FILE}. Returning empty list.")
        return []
    except Exception as e:
        logger.error(f"Unexpected error loading {JSON_FILE}: {e}")
        return []

def save_users(users):
    """Saves user data to the JSON file."""
    if not isinstance(users, list):
        logger.error("Attempted to save non-list data to users file.")
        return False
    try:
        with open(JSON_FILE, 'w', encoding='utf-8') as f:
            json.dump(users, f, indent=4, ensure_ascii=False)
        logger.info(f"Users saved successfully to {JSON_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving users to {JSON_FILE}: {e}")
        return False

@app.route('/users', methods=['GET'])
def get_users():
    """API endpoint to get the list of users."""
    users = load_users()
    return jsonify(users)

@app.route('/users', methods=['POST'])
def update_users():
    """API endpoint to update the list of users."""
    try:
        new_users_data = request.json
        if not isinstance(new_users_data, list):
            logger.error("Received data for saving is not a list.")
            return jsonify({"error": "Invalid data format: Expected a list."}), 400

        required_keys = ('name', 'age', 'profession', 'encoding')
        validated_users = []
        for user in new_users_data:
            if isinstance(user, dict) and all(k in user for k in required_keys):
                validated_users.append(user)
            else:
                logger.warning(f"Invalid user data received, skipping: {user}. Expected keys: {required_keys}")

        if save_users(validated_users):
             logger.info(f"{len(validated_users)} users updated.")
             return jsonify({"message": "Users updated successfully"})
        else:
             logger.error("Failed to save users to file.")
             return jsonify({"error": "Failed to save users"}), 500
    except Exception as e:
        logger.exception("Unexpected error in POST /users endpoint")
        return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    port = 5000
    host = '127.0.0.1'
    logger.info(f"Starting Flask admin server at http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
