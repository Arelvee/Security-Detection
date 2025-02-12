import os
import cv2
import logging
import threading
from flask import Flask, Response
from inference import get_model
import supervision as sv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Load API key from environment variable
API_KEY = os.getenv("MODEL_API_KEY")
if not API_KEY:
    logging.error("API key not found. Set the MODEL_API_KEY environment variable.")
    exit(1)

# Load YOLOv8 model
try:
    model = get_model(model_id="security-and-weapon-detection-d4ya2-yyieq-nuyqk-v1n7e/1", api_key=API_KEY)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

# Initialize camera
def init_camera():
    cap = None
    for i in range(3):  # Try multiple indexes (0, 1, 2)
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            logging.info(f"Camera initialized at index {i}")
            return cap
    logging.error("No camera found!")
    exit(1)

camera = init_camera()

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Lock for thread safety
frame_lock = threading.Lock()
latest_frame = None

def capture_frames():
    global latest_frame
    while True:
        ret, frame = camera.read()
        if not ret:
            logging.error("Unable to read frame from the camera.")
            break
        with frame_lock:
            latest_frame = frame

# Start a separate thread for capturing frames
threading.Thread(target=capture_frames, daemon=True).start()

def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame.copy() if latest_frame is not None else None
        if frame is None:
            continue
        
        try:
            # Run inference on the frame
            results = model.infer(frame)[0]
            detections = sv.Detections.from_inference(results)
            
            # Annotate the frame
            annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)
            
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            logging.error(f"Error processing frame: {e}")

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
