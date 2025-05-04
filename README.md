# 🧠 Real-Time Facial Recognition with Interactive Dashboard

**Connect with me:** [LinkedIn](https://www.linkedin.com/in/coelhoxyz/)

A simplified real-time facial recognition system using OpenCV and Dlib.

If you have suggestions or find issues, please open an issue on GitHub or send me a message on LinkedIn!

## 🚀 How to Run (Simplified)

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/facial-recognition.git # Replace with your repo URL
    cd facial-recognition
    ```

2.  **Set Up Environment & Install Dependencies:**
    (Optional but recommended: Use a virtual environment)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Install requirements:
    ```bash
    pip install -r backend/requirements.txt
    ```
    > **Note:** `dlib` installation might require `cmake` and C++ build tools. See [dlib](http://dlib.net/) or [face_recognition](https://github.com/ageitgey/face_recognition#installation) docs if you have issues.

3.  **Download Models:**
    Download these files, decompress them (`.bz2` -> `.dat`), and place the `.dat` files directly inside the `backend/` directory:
    -   [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
    -   [dlib_face_recognition_resnet_model_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

4.  **Register Faces (Optional but needed for recognition):**
    ```bash
    cd backend
    python src/save_users.py
    ```
    Follow the prompts to add users. Press `c` to capture, `q` to quit.

5.  **Run the Main Server:**
    Make sure you are in the `backend` directory.
    ```bash
    python src/server.py
    ```
    This starts the facial recognition via webcam and the WebSocket server. Press `q` in the OpenCV window to stop.

6.  **Run the Admin Panel (Optional):**
    Open a *new* terminal, navigate to `backend`, activate the environment, and run:
    ```bash
    python admin/admin_server.py
    ```
    Access it at [http://localhost:5000](http://localhost:5000).

## 🎯 Features Overview

-   Real-time face detection and recognition via webcam.
-   Visual feedback in an OpenCV window (bounding boxes, info).
-   WebSocket server for broadcasting results.
-   Basic web admin panel for user management.
-   Logging.

## 🧰 Technologies Used

-   Python 3.8+
-   OpenCV, dlib, face_recognition, NumPy
-   WebSockets, Flask, Flask-CORS

## 📂 Project Structure

```text
facial-recognition/
├── backend/
│   ├── src/
│   │   ├── __init__.py       # Makes src a package (optional but good practice)
│   │   ├── server.py           # Main script: Real-time recognition, WebSocket server, Web UI
│   │   └── utils/
│   │       ├── __init__.py   # Makes utils a sub-package
│   │       └── logger.py   # Logging configuration utility
│   │   └── data/
│   │       └── face_data.json # Stores user data (name, age, profession, encoding)
│   ├── admin/
│   │   ├── __init__.py       # Makes admin a package
│   │   ├── admin_server.py   # Flask API for managing users
│   │   ├── index.html        # Admin panel interface (HTML)
│   │   └── styles.css        # Admin panel CSS
│   ├── shape_predictor_68_face_landmarks.dat   # dlib model (place here)
│   ├── dlib_face_recognition_resnet_model_v1.dat # dlib model (place here)
│   ├── Dockerfile.recognition  # Dockerfile for the main server
│   ├── requirements.txt      # Python dependencies for the backend
│   └── .gitignore            # Git ignore rules for backend
├── .dockerignore             # Docker ignore rules for build context
├── docker-compose.yml        # Docker Compose configuration
└── README.md                 # This file
```

_Note: Model files (`.dat`) should reside directly within the `backend/` directory. User data (`face_data.json`) is expected in `backend/src/data/`._

---