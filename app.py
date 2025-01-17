from flask import Flask, Response
import cv2

app = Flask(__name__)

# Function to capture frames from the laptop camera
def generate_frames():
    camera = cv2.VideoCapture(0)  # 0 is the ID for the laptop's default camera
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Encode the frame as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
