import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import pickle
from tensorflow.keras.models import load_model
from PIL import Image
import io

# --- Page Configuration ---
st.set_page_config(
    page_title="ASL Recognition System",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #E3F2FD;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 10px 0;
    }
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .confidence-low {
        color: #FF9800;
        font-weight: bold;
    }
    .stButton>button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
        font-size: 1.2rem;
        padding: 0.5rem;
        border-radius: 10px;
    }
    </style>
""",
    unsafe_allow_html=True,
)


# --- Load Model Function ---
@st.cache_resource
def load_asl_model(model_path, classes_path):
    """Load the trained model and class names"""
    try:
        model = load_model(model_path)
        with open(classes_path, "rb") as f:
            class_names = pickle.load(f)
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# --- Initialize MediaPipe ---
def get_mediapipe_hands():
    """Get a fresh MediaPipe Hands instance"""
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7,
    )
    return mp_hands, hands


# --- Extract Landmarks ---
def extract_landmarks_from_frame(hand_landmarks):
    """Extract 63 features from detected hand"""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords)


# --- Process Single Image ---
def process_image(image, model, class_names):
    """Process a single image and return predictions"""
    # Initialize MediaPipe for this image
    mp_hands, hands = get_mediapipe_hands()
    mp_drawing = mp.solutions.drawing_utils
    
    try:
        # Convert PIL to numpy array if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Make a copy for drawing
        output_image = image.copy()
        
        # Convert to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        predicted_class = "NOTHING"
        confidence = 0.0
        top3_predictions = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    output_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
                )

                # Extract landmarks for prediction
                landmarks = extract_landmarks_from_frame(hand_landmarks)
                landmarks = landmarks.reshape(1, -1)

                # Predict
                predictions = model.predict(landmarks, verbose=0)
                predicted_idx = np.argmax(predictions[0])
                confidence = predictions[0][predicted_idx]
                predicted_class = class_names[predicted_idx]

                # Get top 3 predictions
                top3_indices = np.argsort(predictions[0])[-3:][::-1]
                top3_predictions = [
                    (class_names[idx], predictions[0][idx]) for idx in top3_indices
                ]

                # Display prediction on image
                text = f"{predicted_class}: {confidence*100:.1f}%"
                color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)
                cv2.putText(output_image, text, (15, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        else:
            # No hand detected
            cv2.putText(output_image, "No hand detected", (15, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        return output_image, predicted_class, confidence, top3_predictions
    
    finally:
        # Always close hands properly
        hands.close()


# --- Main App ---
def main():
    # Header
    st.markdown(
        '<p class="main-header">🤟 ASL Recognition System</p>', unsafe_allow_html=True
    )
    st.markdown(
        '<p class="sub-header">Real-time American Sign Language Recognition using AI</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")

        # Model paths
        st.subheader("📁 Model Configuration")
        model_path = st.text_input(
            "Model Path",
            value="models/asl_landmarks_final.h5",
            help="Path to the trained .h5 model file",
        )

        classes_path = st.text_input(
            "Classes Path",
            value="models/asl_landmarks_classes.pkl",
            help="Path to the classes pickle file",
        )

        st.markdown("---")
        st.markdown("### 📖 Instructions")
        st.markdown(
            """
        1. Load the model using the button below
        2. Use your camera or upload an image
        3. Show ASL hand signs
        4. View predictions in real-time
        """
        )

    # Load model button
    if st.sidebar.button("🔄 Load Model", use_container_width=True):
        with st.spinner("Loading model..."):
            model, class_names = load_asl_model(model_path, classes_path)
            if model is not None:
                st.session_state.model = model
                st.session_state.class_names = class_names
                st.session_state.model_loaded = True
                st.sidebar.success(f"✅ Model loaded! Classes: {len(class_names)}")
            else:
                st.sidebar.error("❌ Failed to load model.")

    # Initialize session state
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False

    # Main content
    if not st.session_state.model_loaded:
        st.warning("⚠️ Please load the model first using the sidebar.")
        return

    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📸 Camera Input", "📤 Upload Image"])

    with tab1:
        st.subheader("📸 Use Your Camera")
        st.info("👉 Take a photo showing an ASL hand sign")
        
        camera_photo = st.camera_input("Take a picture")
        
        if camera_photo is not None:
            # Read image
            image = Image.open(camera_photo)
            image_np = np.array(image)
            
            # Process image
            with st.spinner("Processing..."):
                output_image, predicted_class, confidence, top3_predictions = process_image(
                    image_np,
                    st.session_state.model,
                    st.session_state.class_names
                )
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(output_image, caption="Processed Image", use_column_width=True)
            
            with col2:
                if predicted_class != "NOTHING":
                    conf_class = "confidence-high" if confidence > 0.7 else "confidence-low"
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <h2 style="margin:0; color: #1E88E5;">🤟 {predicted_class}</h2>
                            <p style="font-size: 1.5rem; margin:10px 0 0 0;" class="{conf_class}">
                                Confidence: {confidence*100:.1f}%
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    if top3_predictions:
                        st.markdown("**Top 3 Predictions:**")
                        for i, (label, conf) in enumerate(top3_predictions, 1):
                            st.write(f"{i}. **{label}**: {conf*100:.1f}%")
                else:
                    st.info("👋 No hand detected - Try again!")

    with tab2:
        st.subheader("📤 Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image with an ASL hand sign",
            type=["jpg", "jpeg", "png"]
        )
        
        if uploaded_file is not None:
            # Read image
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Process image
            with st.spinner("Processing..."):
                output_image, predicted_class, confidence, top3_predictions = process_image(
                    image_np,
                    st.session_state.model,
                    st.session_state.class_names
                )
            
            # Display results
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(output_image, caption="Processed Image", use_column_width=True)
            
            with col2:
                if predicted_class != "NOTHING":
                    conf_class = "confidence-high" if confidence > 0.7 else "confidence-low"
                    st.markdown(
                        f"""
                        <div class="prediction-box">
                            <h2 style="margin:0; color: #1E88E5;">🤟 {predicted_class}</h2>
                            <p style="font-size: 1.5rem; margin:10px 0 0 0;" class="{conf_class}">
                                Confidence: {confidence*100:.1f}%
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    if top3_predictions:
                        st.markdown("**Top 3 Predictions:**")
                        for i, (label, conf) in enumerate(top3_predictions, 1):
                            st.write(f"{i}. **{label}**: {conf*100:.1f}%")
                else:
                    st.info("👋 No hand detected - Try uploading another image!")


if __name__ == "__main__":
    main()
