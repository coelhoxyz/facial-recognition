# 🧠 Real-Time Facial Recognition with Interactive Dashboard

A real-time facial recognition system using OpenCV and Dlib, featuring status indicators and an administrative panel for user management. Suitable for access control prototypes, AI demonstrations, or computer vision studies.

## 🎯 Features

-   📸 Real-time face detection via webcam.
-   🔍 Recognition of registered users.
-   🖥️ OpenCV window displaying:
    -   Animated bounding box around detected faces (Color-coded: Green for Granted, Red for Denied, Yellow for Analyzing).
    -   Information balloon next to the face (Name, Age, Profession).
    -   Overall status banner at the top (✔ Access Granted | ✖ Access Denied | 🔎 Analyzing).
-   🌐 WebSocket server broadcasting recognition data (for potential web clients).
-   🛠️ Web-based administrative panel (Flask) to view/edit/delete registered users.
-   📝 Centralized logging for monitoring and debugging.

---

## 🧰 Technologies Used

-   Python 3.8+
-   OpenCV (`opencv-python`)
-   dlib
-   face_recognition
-   NumPy
-   WebSockets (`websockets`)
-   Flask (for the admin dashboard)
-   Flask-CORS

---

## 📂 Project Structure

```
facial-recognition/
├── backend/
│   ├── src/
│   │   ├── server.py           # Main script: Real-time recognition & WebSocket server
│   │   ├── save_users.py       # Script to register new faces via webcam
│   │   └── utils/
│   │       └── logger.py       # Logging configuration utility
│   ├── admin/
│   │   ├── admin_server.py   # Flask API for managing users
│   │   ├── index.html        # Admin panel interface (HTML)
│   │   └── styles.css        # Admin panel CSS
│   ├── face_data.json        # Stores user data (name, age, profession, encoding)
│   ├── shape_predictor_68_face_landmarks.dat   # dlib model (place here)
│   └── dlib_face_recognition_resnet_model_v1.dat # dlib model (place here)
│   └── requirements.txt      # Python dependencies for the backend
└── README.md                 # This file
```

_Note: Model files (`.dat`) and `face_data.json` should reside directly within the `backend/` directory._

---

## 🚀 How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/facial-recognition.git # Replace with your repo URL
cd facial-recognition
```

### 2. Set Up Environment & Install Dependencies

It's recommended to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

Install the required Python packages:

```bash
pip install -r backend/requirements.txt
```

> ⚠️ **dlib Installation:** If you encounter errors installing `dlib`, you might need to install `cmake` and potentially C++ build tools first. Refer to the [official dlib installation guide](http://dlib.net/) or the [face_recognition prerequisites](https://github.com/ageitgey/face_recognition#installation) for detailed instructions specific to your operating system.

### 3. Download Required Models

Download the following pre-trained dlib models:

-   [shape_predictor_68_face_landmarks.dat.bz2](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
-   [dlib_face_recognition_resnet_model_v1.dat.bz2](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

**Decompress** the downloaded `.bz2` files (e.g., using `bunzip2` on Linux/macOS or 7-Zip on Windows) and **move the resulting `.dat` files** into the `backend/` directory.

### 4. Register Your Face

Run the registration script from the `backend` directory:

```bash
cd backend
python src/save_users.py
```

Enter your details (Name, Age, Profession) when prompted. Look at the camera, ensure your face is clear, and press `c` to capture the image. Press `q` to quit.

### 5. Start the Facial Recognition Server

Run the main server script from the `backend` directory:

```bash
# Make sure you are in the 'backend' directory
python src/server.py
```

An OpenCV window titled "Facial Recognition" should appear, showing the camera feed with detection and recognition overlays. The WebSocket server will also start (default: `ws://localhost:8765`). Press `q` in the OpenCV window to stop the server.

### 6. (Optional) Run the Admin Panel

Open a *new* terminal, navigate to the `backend` directory, activate the virtual environment, and run the Flask admin server:

```bash
cd backend
source ../venv/bin/activate # Or ..\venv\Scripts\activate on Windows
python admin/admin_server.py
```

Open [http://localhost:5000](http://localhost:5000) (or the address shown in the terminal) in your web browser to view, edit, or delete registered users.

---

## 💾 Data Storage

User information and facial encodings are stored in the `backend/face_data.json` file. Each entry includes:

-   `name` (String)
-   `age` (String)
-   `profession` (String)
-   `encoding` (List of Floats - the facial features vector)

---

## 💡 Possible Improvements

-   **Database Integration:** Replace `face_data.json` with a proper database (e.g., PostgreSQL, SQLite, MongoDB) for better scalability and data management.
-   **Web Client:** Develop a web-based client that connects to the WebSocket server (`ws://localhost:8765`) to display the recognition status and video feed remotely.
-   **Authentication:** Add user login and permissions to the admin panel.
-   **Configuration File:** Move settings like camera index, WebSocket port, model paths, etc., to a configuration file (e.g., YAML, `.env`).
-   **Deployment:** Containerize the application using Docker for easier deployment.
-   **Performance:** Experiment with different face detection models (`hog` vs `cnn` in `face_recognition`) or frame skipping for performance tuning.
-   **Error Handling:** Implement more robust error handling and user feedback, especially during registration.
