# Sign Language Translator

A real-time American Sign Language (ASL) alphabet translator application that uses Computer Vision and Deep Learning to interpret hand gestures into text. The system consists of a Python-based backend server (FastAPI) and a modern web-based frontend.

## 🚀 Features

- **Real-time Translation:** Instantly translates hand gestures into text using a webcam.
- **Sentence Builder:** Construct full sentences with features to add spaces and delete characters.
- **Client-Server Architecture:** Decoupled frontend and backend communicating via WebSockets.
- **Advanced Computer Vision:** Utilizes MediaPipe for robust hand landmark detection.
- **Deep Learning Model:** Powered by a TensorFlow/Keras model trained on ASL landmarks.
- **Responsive UI:** Modern, user-friendly interface built with Tailwind CSS.

## 🛠️ Tech Stack

### Backend
- **Language:** Python 3.12+
- **Framework:** FastAPI (with WebSockets)
- **ML/AI:** TensorFlow/Keras, MediaPipe, OpenCV, Scikit-learn
- **Data Processing:** NumPy, Pandas

### Frontend
- **Core:** HTML5, Vanilla JavaScript
- **Styling:** Tailwind CSS (via CDN)
- **Communication:** WebSockets

## 📂 Project Structure

```
.
├── app/
│   └── server.py          # Main FastAPI server and WebSocket handler
├── Models/
│   ├── asl_landmarks_final.h5   # Trained Keras model
│   └── asl_landmarks_classes.pkl # Class labels encoder
├── notebooks/
│   └── Copy_of_DEPI_Project.ipynb # Model training notebook
├── index.html             # Frontend client application
├── pyproject.toml         # Project configuration and dependencies
├── uv.lock                # Dependency lock file
└── README.md              # Project documentation
```

## 📋 Prerequisites

- Python 3.12 or higher
- A webcam
- [uv](https://github.com/astral-sh/uv) (Recommended for dependency management) OR `pip`

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  **Install Dependencies:**

    **Using `uv` (Recommended):**
    ```bash
    uv sync
    ```

    **Using `pip`:**
    ```bash
    pip install fastapi[standard] uvicorn tensorflow opencv-python mediapipe numpy pandas scikit-learn
    ```

## 🚀 Usage

### 1. Start the Backend Server

Run the server using Python. Make sure you are in the project root directory.

**Using `uv`:**
```bash
uv run app/server.py
```

**Using standard Python:**
```bash
python app/server.py
```

The server will start on `http://0.0.0.0:8080`.

### 2. Configure the Frontend

1.  Open `index.html` in a text editor.
2.  Locate the `WEBSOCKET_URL` constant (around line 93).
3.  Update it to point to your local server:
    ```javascript
    const WEBSOCKET_URL = "ws://localhost:8080/sign-model";
    ```
    *(Note: The default value might be set to an ngrok URL, which won't work for local testing unless you are tunneling.)*

### 3. Run the Application

1.  Open `index.html` in your web browser.
2.  Allow camera access when prompted.
3.  Wait for the connection status to turn **Green** ("Connected").
4.  Start signing! The predicted characters will appear on the screen, and you can build sentences.

## 🧠 Model Details

The system uses a custom-trained Neural Network that processes hand landmarks extracted by MediaPipe.
- **Input:** 63 coordinates (x, y, z) for 21 hand landmarks.
- **Output:** Classification of the ASL alphabet character.
- **Smoothing:** Predictions are smoothed over a 5-frame window to reduce jitter.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

- Built by:

  - Samy Adel
  - Mahmoud Elsherbiny
  - Mazen Arafat
  - Ahmed Salah
  - Fouad Ramzy
  - Youssef Mustafa
  - Kirollos Safwat

- Uses open-source libraries: TensorFlow, FastAPI, MediaPipe, etc.
