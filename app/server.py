import base64
import sys
import pickle
import numpy as np

# Third-party libraries
try:
    from tensorflow.keras.models import load_model
    import cv2
    import mediapipe as mp
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    import uvicorn
    print("All necessary libraries imported successfully.")
except ImportError as e:
    print(f"Error: A required library is missing. Install with 'pip install fastapi uvicorn tensorflow opencv-python mediapipe'. Error: {e}", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---
HOST = "0.0.0.0" 
PORT = 8080
MODEL_PATH = r"Models/asl_landmarks_final.h5"
CLASSES_PATH = r"Models/asl_landmarks_classes.pkl"
HISTORY_SIZE = 5 # For smoothing predictions over frames

# --- Initialize ML Resources (Global, loaded once) ---
print("Loading Model and Classes...")
try:
    model = load_model(MODEL_PATH)
    with open(CLASSES_PATH, 'rb') as f:
        class_names = pickle.load(f)
    print(f"Model loaded successfully. Found {len(class_names)} classes.")
except Exception as e:
    print(f"Error loading model resources. Check paths and file integrity: {e}", file=sys.stderr)
    sys.exit(1)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# --- FastAPI Application Instance ---
app = FastAPI(title="ASL Real-time WebSocket Server")

# --- Utility Functions (Kept the same) ---

def extract_landmarks(hand_landmarks):
    """Extract 63 features (x, y, z) from detected hand landmarks."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords).reshape(1, -1) # Shape: (1, 63)

def process_frame_for_prediction(frame, prediction_history):
    """
    Processes a single frame, detects hand, predicts sign, and updates history.
    Returns the predicted character or 'NOTHING'.
    """
    # 1. Convert to RGB for MediaPipe
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Process with MediaPipe
    results = hands.process(image_rgb)
    
    predicted_class = "NOTHING"
    
    if results.multi_hand_landmarks:
        # Assuming only one hand is processed (max_num_hands=1)
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # 3. Extract landmarks
        landmarks = extract_landmarks(hand_landmarks)
        
        # 4. Predict
        predictions = model.predict(landmarks, verbose=0)
        
        # --- Smoothing Logic ---
        # Note: We can simplify the smoothing logic slightly here if desired, 
        # but sticking to the current index-based history for consistency.
        predicted_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_idx]
        
        prediction_history.append(predicted_idx)
        if len(prediction_history) > HISTORY_SIZE:
            prediction_history.pop(0)
            
        # Use most common prediction
        if prediction_history:
            final_prediction_idx = max(set(prediction_history), key=prediction_history.count)
            predicted_class = class_names[final_prediction_idx]
            
        print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
        
    else:
        # No hand detected, clear history
        if prediction_history:
             prediction_history.clear()
        print("Prediction: NOTHING (No hand detected)")

    return predicted_class

# --- FastAPI WebSocket Endpoint ---

@app.websocket("/sign-model")
async def websocket_endpoint(websocket: WebSocket):
    """Handles incoming WebSocket connections and frame processing."""
    await websocket.accept()
    client_host = websocket.client.host
    print(f"New connection established: {client_host}")
    
    # History is specific to this connection
    prediction_history = [] 

    try:
        while True:
            # 1. Receive Base64 text frame from the client
            message = await websocket.receive_text()
            
            # 2. Decode Base64 string to bytes and convert to OpenCV image
            try:
                # The client sends the raw base64 data (after the comma in the data URL)
                img_data = base64.b64decode(message)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            except Exception as decode_error:
                print(f"Error decoding frame from client {client_host}: {decode_error}", file=sys.stderr)
                continue # Skip to next frame

            if frame is not None:
                # 3. Process the frame and get the prediction
                prediction = process_frame_for_prediction(frame, prediction_history)
                
                # 4. Send the prediction back to the client
                await websocket.send_text(prediction)
            
    except WebSocketDisconnect:
        print(f"Connection closed by client: {client_host}")
    except Exception as e:
        print(f"An error occurred in the connection handler for {client_host}: {e}", file=sys.stderr)
    finally:
        print(f"Handler finished for {client_host}")

# --- Main Server Run Block ---

if __name__ == "__main__":
    # Note: The reload=True argument is useful during development
    print(f"Starting FastAPI server using Uvicorn on http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)