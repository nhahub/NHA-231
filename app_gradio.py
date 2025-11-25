import gradio as gr
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
import time
from collections import deque

# --- Global Variables ---
model = None
class_names = None
mp_hands = None
mp_drawing = None
hands = None
prediction_history = deque(maxlen=5)
is_running = False
cap = None

# --- Custom CSS ---
custom_css = """
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E88E5;
    text-align: center;
    margin-bottom: 1rem;
}
.prediction-box {
    background-color: #E3F2FD;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #1E88E5;
    margin: 10px 0;
    text-align: center;
}
.confidence-high {
    color: #4CAF50;
    font-weight: bold;
    font-size: 1.5rem;
}
.confidence-low {
    color: #FF9800;
    font-weight: bold;
    font-size: 1.5rem;
}
.stat-box {
    background-color: #f5f5f5;
    padding: 15px;
    border-radius: 8px;
    text-align: center;
    margin: 5px;
}
"""

# --- Load Model Function ---


def load_asl_model(model_path, classes_path):
    """Load the trained model and class names"""
    global model, class_names
    try:
        model = load_model(model_path)
        with open(classes_path, "rb") as f:
            class_names = pickle.load(f)
        return f"✅ Model loaded successfully! Classes: {len(class_names)}", gr.update(interactive=True)
    except Exception as e:
        return f"❌ Error loading model: {str(e)}", gr.update(interactive=False)


# --- Initialize MediaPipe ---
def init_mediapipe():
    """Initialize MediaPipe Hands"""
    global mp_hands, mp_drawing, hands
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )


# --- Extract Landmarks ---
def extract_landmarks_from_frame(hand_landmarks):
    """Extract 63 features from detected hand"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords)


# --- Process Frame ---
def process_single_frame(frame, history_size=5):
    """Process a single frame and return predictions"""
    global model, class_names, mp_hands, mp_drawing, hands, prediction_history

    if model is None or hands is None:
        return frame, "⚠️ Model not loaded", "", ""

    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape

    # Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    predicted_class = "NOTHING"
    confidence = 0.0
    top3_predictions = []
    prediction_html = ""
    top3_html = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
            )

            # Extract landmarks for prediction
            landmarks = extract_landmarks_from_frame(hand_landmarks)
            landmarks = landmarks.reshape(1, -1)

            # Predict
            predictions = model.predict(landmarks, verbose=0)
            predicted_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_idx]

            # Smooth predictions
            prediction_history.append(predicted_idx)

            # Use most common prediction
            if len(prediction_history) > 0:
                final_prediction = max(
                    set(prediction_history), key=list(prediction_history).count)
                predicted_class = class_names[final_prediction]

            # Get top 3 predictions
            top3_indices = np.argsort(predictions[0])[-3:][::-1]
            top3_predictions = [
                (class_names[idx], predictions[0][idx]) for idx in top3_indices
            ]

            # Display prediction on frame
            text = f"{predicted_class}: {confidence*100:.1f}%"
            text_size = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            cv2.rectangle(frame, (10, 10),
                          (20 + text_size[0], 60), (0, 0, 0), -1)

            color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
            cv2.putText(frame, text, (15, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

            # Create prediction HTML
            conf_class = "confidence-high" if confidence > 0.7 else "confidence-low"
            prediction_html = f"""
            <div class="prediction-box">
                <h2 style="margin:0; color: #1E88E5;">🤟 {predicted_class}</h2>
                <p class="{conf_class}" style="margin:10px 0 0 0;">
                    Confidence: {confidence*100:.1f}%
                </p>
            </div>
            """

            # Create top 3 HTML
            top3_html = "<h3>📊 Top 3 Predictions:</h3>"
            for i, (label, conf) in enumerate(top3_predictions, 1):
                top3_html += f"<p><strong>{i}. {label}:</strong> {conf*100:.1f}%</p>"

    else:
        # No hand detected
        cv2.putText(frame, "NOTHING", (15, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, "No hand detected", (15, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        prediction_history.clear()
        prediction_html = "<div class='prediction-box'><h3>👋 No hand detected - Show a hand sign!</h3></div>"
        top3_html = ""

    # Convert BGR to RGB for display
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame_rgb, prediction_html, top3_html


# --- Video Stream Generator ---
def video_stream(history_size):
    """Generator for video stream"""
    global cap, is_running

    if cap is None or not cap.isOpened():
        yield None, "<p style='color: red;'>❌ Camera not initialized</p>", "", ""
        return

    frame_count = 0
    start_time = time.time()

    while is_running:
        ret, frame = cap.read()
        if not ret:
            yield None, "<p style='color: red;'>❌ Failed to grab frame</p>", "", ""
            break

        # Process frame
        processed_frame, prediction_html, top3_html = process_single_frame(
            frame, history_size)

        # Calculate stats
        frame_count += 1
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time if elapsed_time > 0 else 0

        stats_html = f"""
        <div style="display: flex; gap: 10px; justify-content: center; margin-top: 10px;">
            <div class="stat-box">
                <strong>FPS</strong><br>{fps:.1f}
            </div>
            <div class="stat-box">
                <strong>Frames</strong><br>{frame_count}
            </div>
        </div>
        """

        yield processed_frame, prediction_html, top3_html, stats_html
        time.sleep(0.03)


# --- Start Camera ---
def start_camera(camera_index):
    """Start the camera stream"""
    global cap, is_running, hands

    if model is None:
        return None, "<p style='color: orange;'>⚠️ Please load the model first!</p>", "", "", gr.update(interactive=False), gr.update(interactive=True)

    # Initialize MediaPipe if not done
    if hands is None:
        init_mediapipe()

    # Open camera
    cap = cv2.VideoCapture(int(camera_index))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        return None, "<p style='color: red;'>❌ Failed to open camera</p>", "", "", gr.update(interactive=True), gr.update(interactive=False)

    is_running = True
    return None, "<p style='color: green;'>🎥 Camera is running...</p>", "", "", gr.update(interactive=False), gr.update(interactive=True)


# --- Stop Camera ---
def stop_camera():
    """Stop the camera stream"""
    global cap, is_running, prediction_history, hands

    is_running = False

    if cap is not None:
        cap.release()
        cap = None

    if hands is not None:
        hands.close()
        hands = None

    prediction_history.clear()

    return None, "📷 Camera stopped", "", "", gr.update(interactive=True), gr.update(interactive=False)


# --- Build Gradio Interface ---
def build_interface():
    with gr.Blocks(title="ASL Recognition System") as demo:
        gr.HTML(f"<style>{custom_css}</style>")
        # Header
        gr.HTML("""
            <div class="main-header">
                🤟 ASL Recognition System
            </div>
            <p style="text-align: center; font-size: 1.2rem; color: #424242; margin-bottom: 2rem;">
                Real-time American Sign Language Recognition using AI
            </p>
        """)

        with gr.Row():
            # Left Column - Settings
            with gr.Column(scale=1):
                gr.Markdown("## ⚙️ Settings")

                gr.Markdown("### 🔧 Model Configuration")
                model_path = gr.Textbox(
                    label="Model Path",
                    value="models/asl_landmarks_final.h5",
                    info="Path to the trained .h5 model file"
                )

                classes_path = gr.Textbox(
                    label="Classes Path",
                    value="models/asl_landmarks_classes.pkl",
                    info="Path to the classes pickle file"
                )

                load_btn = gr.Button(
                    "🔄 Load Model", variant="primary", size="lg")
                load_status = gr.HTML("")

                gr.Markdown("### 📹 Camera Settings")
                camera_index = gr.Slider(
                    label="Camera Index",
                    minimum=0,
                    maximum=5,
                    value=0,
                    step=1
                )

                gr.Markdown("### 🎯 Detection Settings")
                history_size = gr.Slider(
                    label="Prediction Smoothing",
                    minimum=1,
                    maximum=10,
                    value=5,
                    step=1,
                    info="Number of frames to smooth predictions"
                )

                gr.Markdown("---")
                gr.Markdown("""
                ### 📖 Instructions
                1. Load the model using the button above
                2. Click **Start Camera** to begin
                3. Show ASL hand signs to the camera
                4. View real-time predictions
                5. Click **Stop Camera** when done
                """)

            # Right Column - Video and Results
            with gr.Column(scale=2):
                gr.Markdown("## 📹 Live Camera Feed")

                with gr.Row():
                    start_btn = gr.Button(
                        "▶️ Start Camera", variant="primary", size="lg", scale=1)
                    stop_btn = gr.Button(
                        "⏹️ Stop Camera", variant="stop", size="lg", scale=1, interactive=False)

                camera_status = gr.HTML(
                    "<p style='text-align: center;'>📷 Camera is ready</p>")

                video_output = gr.Image(
                    label="Camera Feed",
                    type="numpy",
                    height=500,
                    show_label=False
                )

                gr.HTML("<br>")  # مسافة إضافية

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("## 📊 Current Prediction")
                        prediction_output = gr.HTML("")

                    with gr.Column(scale=1):
                        top3_output = gr.HTML("")

                stats_output = gr.HTML("")

        # Event Handlers
        load_btn.click(
            fn=load_asl_model,
            inputs=[model_path, classes_path],
            outputs=[load_status, start_btn]
        )

        start_btn.click(
            fn=start_camera,
            inputs=[camera_index],
            outputs=[video_output, camera_status,
                     prediction_output, top3_output, start_btn, stop_btn]
        ).then(
            fn=video_stream,
            inputs=[history_size],
            outputs=[video_output, prediction_output,
                     top3_output, stats_output]
        )

        stop_btn.click(
            fn=stop_camera,
            outputs=[video_output, camera_status,
                     prediction_output, top3_output, start_btn, stop_btn]
        )

    return demo


# --- Main ---
if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=True,
        show_error=True
    )
