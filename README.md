
# 🖐️ American Sign Language Recognition using MediaPipe and TensorFlow

This project uses **MediaPipe Hand Landmarks** and a **Neural Network** built with **TensorFlow** to recognize American Sign Language (ASL) alphabet gestures in real time from a webcam feed.

---

## 📚 Overview

The project is divided into two main parts:

1. **Model Training:**
   - Extracts **21 hand landmarks (63 features)** using **MediaPipe Hands** from the ASL Alphabet dataset.
   - Trains a fully connected neural network on these landmarks to classify ASL alphabet gestures.
   - Saves the trained model and label encoder for future inference.

2. **Real-Time Recognition:**
   - Loads the trained model and label classes.
   - Captures frames from a **webcam** and detects hands using MediaPipe.
   - Predicts the corresponding ASL letter and displays it on the screen with live confidence levels.

---

## 🧩 Project Structure

```
📁 asl-recognition/
│
├── 📄 train_asl_model.py        # Training script (from the notebook)
├── 📄 test_webcam_asl.py        # Real-time recognition with webcam
├── 📄 requirements.txt          # Required dependencies
├── 📄 README.md                 # Project documentation (this file)
│
├── 📁 model/
│   ├── asl_landmarks_final.h5          # Trained model file
│   ├── asl_landmarks_classes.pkl       # Saved label encoder classes
│
├── 📁 dataset/
│   ├── asl_alphabet_train/             # Training data (A–Z)
│   └── asl_alphabet_test/              # Test images
```

---

## ⚙️ Installation

1. **Clone or download** this repository.  
2. Install all dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your dataset is in the correct directory structure:
   ```
   /kaggle/input/asl-alphabet/
   ├── asl_alphabet_train/
   └── asl_alphabet_test/
   ```

---

## 🧠 Training the Model

The training script:

- Extracts 21 × 3 = **63 landmark features** per image using **MediaPipe Hands**.
- Trains a simple **fully connected neural network (Dense layers)**.
- Saves the best model as `asl_landmarks_final.h5` and the class labels as `asl_landmarks_classes.pkl`.

Run:

```bash
python train_asl_model.py
```

After training, you should see something like:

```
✅ Model and classes saved successfully!
📊 Final Validation Accuracy: 98.5%
```

---

## 🎥 Real-Time ASL Recognition (Webcam)

Once the model is trained (or downloaded), test it live using your webcam.

### Steps:

1. Open `test_webcam_asl.py`.
2. Update model and class paths if necessary:
   ```python
   model = load_model("path/to/asl_landmarks_final.h5")
   with open("path/to/asl_landmarks_classes.pkl", "rb") as f:
       class_names = pickle.load(f)
   ```
3. Run:
   ```bash
   python test_webcam_asl.py
   ```

4. Press **‘q’** to quit.

---

## 🧠 Model Architecture

| Layer Type        | Units | Activation | Dropout | Notes |
|--------------------|--------|-------------|----------|--------|
| Dense              | 256    | ReLU        | 0.4      | Input (63 features) |
| BatchNormalization | —      | —           | —        | — |
| Dense              | 128    | ReLU        | 0.3      | — |
| Dense              | 64     | ReLU        | 0.2      | — |
| Dense (Output)     | #Classes | Softmax   | —        | Prediction layer |

---

## 📈 Example Results

- Validation Accuracy: **95–99%** depending on dataset quality.
- Real-time FPS: **15–25 FPS** on standard webcams.
- Smooth prediction mechanism using a **rolling average of recent predictions**.

---

## 🧾 Requirements

See [`requirements.txt`](requirements.txt):

```
numpy
pandas
opencv-python
mediapipe
tensorflow
scikit-learn
matplotlib
tqdm
```

---

## 🧍‍♂️ How It Works (Simplified)

1. **MediaPipe Hands** detects the hand and extracts 21 3D landmarks.  
2. The **x, y, z coordinates** are flattened into a 63-dimensional vector.  
3. This vector is fed into a trained **Neural Network** model.  
4. The model predicts the ASL alphabet letter.  
5. Results are displayed in real-time on the video feed.

---

## 💡 Future Improvements

- Support for **two-hand gestures**.
- Integration with **sentence recognition** or **continuous signing**.
- Convert model to **TensorFlow Lite** for mobile deployment.

---

## 🏁 Acknowledgments

- **ASL Alphabet Dataset** from [Kaggle](https://www.kaggle.com/grassknoted/asl-alphabet).
- **MediaPipe** by Google for hand tracking.
- **TensorFlow** for deep learning framework.
