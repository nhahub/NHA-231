# American Sign Language Recognition System

A complete ASL alphabet recognition system that uses MediaPipe Hand Landmarks and a Neural Network built with TensorFlow to recognize American Sign Language gestures in real-time. The project includes both a trained model and an interactive Streamlit web application for building words letter-by-letter using ASL signs.

---

## Overview

This project consists of three main components:

### 1. Model Training
- Extracts 21 hand landmarks (63 features: x, y, z coordinates) using MediaPipe Hands from the ASL Alphabet dataset.
- Trains a fully connected neural network with batch normalization and dropout layers.
- Achieves 99.07% validation accuracy with only 0.0334 validation loss.
- Saves the trained model (`asl_landmarks_final.h5`) and label encoder (`asl_landmarks_classes.pkl`).

### 2. Streamlit Web Application (ASL Word Builder)
- Interactive real-time ASL recognition interface.
- Word building: Hold ASL signs to build complete words letter-by-letter.
- Smart confirmation: Letters are added only after holding the sign for a configurable duration (default: 2 seconds).
- Special gestures: Support for SPACE and DEL (delete) commands.
- Word history: Save and track completed words.
- Visual feedback: Live prediction confidence, progress bars, and hand landmark visualization.
- Customizable settings: Adjustable hold time, cooldown duration, and camera selection.

### 3. Real-Time Recognition (Webcam Script)
- Standalone Python script for quick testing.
- Captures frames from webcam and predicts ASL letters.
- Displays predictions with live confidence levels.

---

## Project Structure
```
asl-recognition/
│
├── app.py                          # Streamlit web application (Word Builder)
├── train_asl_model.py              # Model training script
├── test_webcam_asl.py              # Real-time webcam recognition script
├── requirements.txt                # Required dependencies
├── README.md                       # Project documentation (this file)
│
├── Models/
│   ├── asl_landmarks_final.h5      # Trained model (99.07% accuracy)
│   └── asl_landmarks_classes.pkl   # Saved label encoder classes
│
└── dataset/
    ├── asl_alphabet_train/         # Training data (A-Z)
    └── asl_alphabet_test/          # Test images
```

---

## System Requirements

- **Python Version:** 3.10.0
- **Operating System:** Windows, macOS, or Linux
- **Hardware Requirements:**
  - Webcam for real-time recognition
  - Minimum 4GB RAM
  - CPU: Intel i5 or equivalent (GPU optional for faster training)

---

## Installation

### Step 1: Clone the Repository
```bash
git clone <repository-url>
cd asl-recognition
```

### Step 2: Create Virtual Environment

It is recommended to use a virtual environment to manage dependencies and avoid conflicts with other projects.

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**For macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

Once the virtual environment is activated, install all required packages:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

Verify that all packages are installed correctly:
```bash
pip list
```

### Step 5: Prepare Model Files

Ensure your model files are in the correct directory:
```
Models/
├── asl_landmarks_final.h5
└── asl_landmarks_classes.pkl
```

Update the paths in `app.py` if necessary:
```python
MODEL_PATH = r"Models/asl_landmarks_final.h5"
CLASSES_PATH = r"Models/asl_landmarks_classes.pkl"
```

---

## Requirements

See `requirements.txt`:
```
numpy==1.26.4
opencv-python-headless==4.10.0.84
mediapipe==0.10.14
tensorflow==2.17.0
streamlit==1.39.0
Pillow==10.4.0
pandas==2.2.3
scikit-learn==1.5.2
PyOpenGL==3.1.7
PyOpenGL-accelerate==3.1.7
```

---

## Model Architecture

The neural network model consists of multiple fully connected layers with batch normalization and dropout for regularization.
```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense)                   │ (None, 256)            │        16,384 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 256)            │         1,024 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout: 0.4)          │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 128)            │        32,896 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 128)            │           512 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout: 0.3)        │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 64)             │         8,256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 64)             │           256 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout: 0.2)        │ (None, 64)             │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense - Softmax)       │ (None, 28)             │         1,820 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
```

### Model Specifications

| Layer Type             | Units     | Activation | Dropout | Notes              |
|------------------------|-----------|------------|---------|-------------------|
| Dense                  | 256       | ReLU       | 0.4     | Input (63 features) |
| BatchNormalization     | -         | -          | -       | -                 |
| Dense                  | 128       | ReLU       | 0.3     | -                 |
| BatchNormalization     | -         | -          | -       | -                 |
| Dense                  | 64        | ReLU       | 0.2     | -                 |
| BatchNormalization     | -         | -          | -       | -                 |
| Dense (Output)         | 28        | Softmax    | -       | Prediction layer  |

**Total Parameters:** 61,148 trainable parameters

---

## Hand Landmark Extraction with MediaPipe

This section explains the process of extracting hand landmarks using MediaPipe, which forms the foundation of the gesture recognition system.

### Overview

The function extracts 21 hand landmarks from an input image. Each landmark provides x, y, z coordinates, resulting in 63 features per hand (21 × 3).

### 1. MediaPipe Hand Detection

- MediaPipe detects hands using a deep learning model.
- For each detected hand, it returns 21 key points representing important joints and fingertips.
- Landmarks are normalized relative to the image size:

$$x_{\text{norm}} = \frac{x_{\text{pixel}}}{w}, \quad y_{\text{norm}} = \frac{y_{\text{pixel}}}{h}, \quad z_{\text{norm}} \in [-1, 1]$$

Where:
- $x_{\text{norm}}, y_{\text{norm}}$ → normalized 2D coordinates
- $z_{\text{norm}}$ → depth coordinate (distance from camera), relative to the wrist depth
- $w, h$ → width and height of the input image in pixels

### 2. Converting to Pixel Coordinates

To map the normalized coordinates back to actual image pixels:

$$x_{\text{pixel}} = x_{\text{norm}} \cdot w$$

$$y_{\text{pixel}} = y_{\text{norm}} \cdot h$$

- $x_{\text{pixel}}, y_{\text{pixel}}$ are now in image coordinates.
- The depth $z_{\text{norm}}$ is typically left normalized for pose analysis, as its scale is non-linear and device-dependent.

### 3. Creating the Feature Vector

All 21 landmarks are flattened into a single feature vector:

$$\mathbf{F} = [x_1, y_1, z_1, \; x_2, y_2, z_2, \; \dots, \; x_{21}, y_{21}, z_{21}] \in \mathbb{R}^{63}$$

- This vector encodes the spatial structure of the hand's pose and shape.
- Can be used as input features for gesture classification or other machine learning models.

### 4. Extraction Process

1. Load the image.
2. Convert from BGR to RGB (required by MediaPipe).
3. Detect hands and extract 21 normalized landmarks.
4. Flatten the (x, y, z) coordinates into a 63-dimensional vector.
5. Return the vector, or None if no hand is detected.

This approach captures the hand's shape and pose numerically, enabling accurate gesture recognition.

---

## Training the Model

The training script:

- Extracts 21 × 3 = 63 landmark features per image using MediaPipe Hands.
- Trains a fully connected neural network with batch normalization and dropout.
- Saves the best model as `asl_landmarks_final.h5` and the class labels as `asl_landmarks_classes.pkl`.

### Steps

1. Ensure your dataset is in the correct directory structure:
```
   dataset/
   ├── asl_alphabet_train/
   │   ├── A/
   │   ├── B/
   │   └── ...
   └── asl_alphabet_test/
       ├── A/
       ├── B/
       └── ...
```

2. Run:
```bash
   python train_asl_model.py
```

3. After training, you should see output similar to:
```
   Model and classes saved successfully!
   Final Validation Accuracy: 99.07%
   Final Validation Loss: 0.0334
```

---

## Usage

### Option 1: Streamlit Web Application (Recommended)

Run the interactive word builder application:
```bash
streamlit run app.py
```

#### Features

- Start/Stop Camera - Control camera feed with buttons
- Sign Letters - Hold ASL signs for 2 seconds to add letters
- Build Words - Compose complete words letter-by-letter
- Save Words - Store completed words in history
- Adjust Settings - Customize hold time and cooldown in sidebar
- Live Feedback - See predictions, confidence, and progress in real-time

#### Instructions

1. Click "Start Camera"
2. Position your hand in front of the camera
3. Make an ASL letter sign
4. Hold the sign for 2 seconds (watch the progress bar)
5. The letter will be added to your current word
6. Use SPACE gesture to add spaces
7. Use DEL gesture to delete last character
8. Click "Save" to add the word to history

### Option 2: Standalone Webcam Script

For quick testing without the web interface:
```bash
python test_webcam_asl.py
```

Press 'q' to quit.

---

## Model Performance

### Training Results

- **Final Validation Accuracy:** 99.07%
- **Final Validation Loss:** 0.0334
- **Training Dataset:** ASL Alphabet dataset from Kaggle
- **Number of Classes:** 28 (A-Z + SPACE + DEL)
- **Total Parameters:** 61,148

### Real-Time Performance

- **Frames Per Second:** 15-30 FPS (depending on hardware)
- **Prediction Smoothing:** Rolling average over last 5 frames for stability
- **Detection Confidence Threshold:** 0.7 (70%)
- **Letter Confirmation Time:** 2 seconds (configurable)

---

## How It Works

### Detection Pipeline

1. MediaPipe Hands detects the hand and extracts 21 3D landmarks.
2. The x, y, z coordinates are flattened into a 63-dimensional vector.
3. This vector is fed into the trained Neural Network model.
4. The model predicts the ASL alphabet letter with confidence score.
5. Prediction smoothing uses a rolling average to reduce noise.

### Word Building Logic

1. User holds an ASL sign in front of the camera.
2. System detects hand and predicts the letter continuously.
3. When the same letter is held for 2 seconds (configurable), it is added to the word.
4. Cooldown period prevents duplicate letters.
5. Special gestures (SPACE, DEL) provide editing capabilities.
6. Completed words can be saved to history.

---

## Future Improvements

- Support for two-hand gestures and words.
- Integration with sentence recognition or continuous signing.
- Convert model to TensorFlow Lite for mobile deployment.
- Multi-language support beyond ASL.
- Text-to-speech output for completed words.
- Gesture-based UI navigation.

---

## Acknowledgments

- **ASL Alphabet Dataset** from Kaggle (grassknoted/asl-alphabet).
- **MediaPipe** by Google for hand tracking technology.
- **TensorFlow** for deep learning framework.
- **Streamlit** for the interactive web application framework.

---

## License

This project is open source and available for educational purposes.

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

Please ensure all contributions maintain the existing code quality and include appropriate documentation.

---

Built with Python 3.10.0, Streamlit, MediaPipe, and TensorFlow
