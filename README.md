# Sign Language Translator

A real-time American Sign Language (ASL) alphabet translator application that uses Computer Vision and Deep Learning to interpret hand gestures into text. The system consists of a Python-based backend server (FastAPI) and a modern web-based frontend.

## 🚀 Features

- **Real-time Translation:** Instantly translates hand gestures into text using a webcam.
- **Sentence Builder:** Construct full sentences with features to add spaces and delete characters.
- **Client-Server Architecture:** Decoupled frontend and backend communicating via WebSockets.
- **Advanced Computer Vision:** Utilizes MediaPipe for robust hand landmark detection.
- **Deep Learning Model:** Powered by a TensorFlow/Keras model trained on ASL landmarks.
- **Responsive UI:** Modern and user-friendly interface.

## 🛠️ Tech Stack

### Backend
- **Language:** Python 3.12+
- **Framework:** FastAPI (with WebSockets)
- **ML/AI:** TensorFlow/Keras, MediaPipe, OpenCV, Scikit-learn
- **Data Processing:** NumPy, Pandas

### Frontend
- **Core:** HTML5, Vanilla JavaScript
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

# 🌐 Running the Application Using Ngrok (Tunneling Mode)

If you want the deployed frontend on **GitHub Pages** to communicate with your locally running backend, you must expose your local server using **Ngrok**.
This allows external clients (like your GitHub Pages site) to connect to your machine securely.

### ✔️ 1. Download & Place `ngrok.exe`
1. Download ngrok from the official website:
   https://ngrok.com/download
2. Place the downloaded `ngrok.exe` file **inside the project root directory**, next to:
```text
DEPI_Final/
├─ ngrok.exe
├─ app/
├─ Models/
├─ WebApp/
└─ server.py
```

---

### ✔️ 2. Authenticate Ngrok
Open a terminal in the project root and run:

```bash
./ngrok.exe config add-authtoken <YOUR_NGROK_AUTH_TOKEN>
```
Replace `<YOUR_NGROK_AUTH_TOKEN>` with the token from your Ngrok dashboard.

### ✔️ 3. Start Ngrok Tunnel
Still in the root directory, run:

```bash
./ngrok.exe http 8080
```
Ngrok will generate a public HTTPS/WebSocket-safe URL like:

```text
https://extranuclear-tiara-semimalignantly.ngrok-free.dev
```
Copy the URL shown under Forwarding.

### ✔️ 4. Update the Frontend (WebSocket URL)
In your deployed or local `index.html`, ensure the following line points to your Ngrok tunnel:

```javascript
const WEBSOCKET_URL = "wss://extranuclear-tiara-semimalignantly.ngrok-free.dev/sign-model";
```
**Important:**
Always use `wss://` (secure WebSocket) with Ngrok, not `ws://`.

### ✔️ 5. Start the Backend Server
Open a new terminal window and run the backend while Ngrok is still running:

**Using uv:**

```bash
uv run app/server.py
```

**Or using Python directly:**

```bash
python app/server.py
```

It will start at:

```text
http://0.0.0.0:8080
```

### ✔️ 6. Open the Deployed Frontend (GitHub Pages)
Your frontend is already deployed at:

🔗 https://samyadel123.github.io/DEPI_Final/

**Now:**

1. Open the link
2. Allow camera access
3. Wait for the status indicator to turn **Green** ("Connected")
4. Begin signing — predictions will appear in real-time

### 🎉 You're All Set!
Your GitHub Pages frontend is now successfully connected to your local backend through Ngrok, enabling real-time sign language detection anywhere on the web.


## 🧠 Model Details

The system uses a custom-trained Neural Network that processes hand landmarks extracted by MediaPipe.
- **Input:** 63 coordinates (x, y, z) for 21 hand landmarks.
- **Output:** Classification of the ASL alphabet character.
- **Smoothing:** Predictions are smoothed over a 5-frame window to reduce jitter.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

- Built by:

  - Samy Adel
  - Mahmoud Elsherbiny
  - Mazen Arafat
  - Ahmed Salah
  - Fouad Ramzy
  - Youssef Mustafa
  - Kirollos Safwat

## Team members roles:

### Samy Adel & Ahmed Salah:

- Real-Time Analysis: Investigated why CNN and MobileNetV2 failed in real-time and proposed switching to a MediaPipe + MLP pipeline.
- MLflow Integration: Set up MLflow to track experiments, metrics, and model versions for full reproducibility.
- Deployment Implementation: Built both the frontend (index.html) and backend (server.py) with WebSockets for real-time communication.
- Hosting & Tunneling: Deployed the frontend on GitHub Pages and used ngrok to create a tunnel linking the browser to the backend.
- Future Research: Started experimenting with a MediaPipe + RNN model to support dynamic gesture recognition.

### Mazen Arafat & Fouad Ramzy & Kirollos Safwat

- Applied landmarks approach and trained MLP Model
- Made deployment using streamlit
- Prepared Presentation
- Worked on Report

### Mahmoud Elsherbiny & Youssef Mustafa

- Trained CNN Model
- Trained MobileNetV2 Model
- Made comparison between Models
- Worked on Report


> Uses open-source libraries: TensorFlow, FastAPI, MediaPipe, etc.
