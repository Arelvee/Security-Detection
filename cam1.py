import os
import cv2
import logging
import threading
from datetime import datetime
from flask import Flask, Response, jsonify, send_from_directory
from queue import Queue

from ultralytics import YOLO
import supervision as sv

logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

SAVE_DIR = os.path.join(app.root_path, 'static/detected_images')
os.makedirs(SAVE_DIR, exist_ok=True)

MODEL_PATH = "firearmed.pt"
if not os.path.exists(MODEL_PATH):
    logging.error(f"Model file '{MODEL_PATH}' not found!")
    exit(1)

try:
    model = YOLO(MODEL_PATH)
    logging.info(f"Loaded model from {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    exit(1)

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    logging.error("Camera failed to open. Check if the camera is connected or in use.")
    exit(1)

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

frame_lock = threading.Lock()
latest_frame = None
save_queue = Queue()

def capture_frames():
    """ Continuously captures frames from the camera without interruptions. """
    global latest_frame
    while True:
        success, frame = camera.read()
        if not success:
            logging.error("Failed to read from camera.")
            break

        with frame_lock:
            latest_frame = frame.copy()

thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

def save_detected_image(frame):
    """ Saves detected images asynchronously without interrupting the stream. """
    filename = f"detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    filepath = os.path.join(SAVE_DIR, filename)
    
    success = cv2.imwrite(filepath, frame)
    if success:
        logging.info(f"✅ Auto-saved detected image: {os.path.abspath(filepath)}")
    else:
        logging.error("❌ Failed to auto-save image.")

def save_worker():
    """ Background thread that continuously saves detected images. """
    while True:
        frame = save_queue.get()
        save_detected_image(frame)
        save_queue.task_done()

save_thread = threading.Thread(target=save_worker, daemon=True)
save_thread.start()

def generate_frames():
    """ Streams frames and saves detected images asynchronously without lag. """
    while True:
        with frame_lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        try:
            results = model(frame)
            if results and len(results[0].boxes) > 0:
                detections = sv.Detections.from_ultralytics(results[0])
                annotated_frame = bounding_box_annotator.annotate(scene=frame, detections=detections)
                annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections)

                save_queue.put(annotated_frame.copy())  # ✅ Add to queue without blocking

                display_frame = annotated_frame
            else:
                display_frame = frame

            # ✅ Keep the stream alive without interruptions
            _, buffer = cv2.imencode('.jpg', display_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        except Exception as e:
            logging.error(f"Error processing frame: {e}")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_images', methods=['GET'])
def get_images():
    images = [img for img in os.listdir(SAVE_DIR) if img.endswith(('jpg', 'png', 'jpeg'))]
    logging.info(f"Available images: {images}")
    return jsonify({"images": images})


@app.route('/detected_images/<filename>')
def serve_image(filename):
    return send_from_directory(SAVE_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
