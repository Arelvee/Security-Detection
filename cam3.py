from flask import Flask, Response
from inference import get_model
import supervision as sv
import cv2

app = Flask(__name__)

# Use your laptop camera as the video source
camera = cv2.VideoCapture(0)  # 0 is typically the default camera

# Check if the video stream is successfully opened
if not camera.isOpened():
    print("Error: Unable to access the video source.")
    exit()

# Load a pre-trained YOLOv8 model
model = get_model(model_id="security-and-weapon-detection-d4ya2/3", api_key="WIL8DJQUU5kSYrlNdheE")

# Create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Function to generate frames with YOLOv8 annotations
def generate_frames():
    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Unable to read frame from the video source.")
            break

        # Run inference on the frame
        results = model.infer(frame)[0]

        # Load the results into the supervision Detections API
        detections = sv.Detections.from_inference(results)

        # Annotate the frame with inference results
        annotated_frame = bounding_box_annotator.annotate(
            scene=frame, detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections)

        # Encode the annotated frame as a JPEG
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
