import cv2
import base64
from ultralytics import YOLO, solutions
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app)

# Load YOLO model
model = YOLO('yolov8n.pt')
classes_to_count = [0]  # 0 for person

# Set up the counter with a horizontal line
frame_width = 1280  # Adjust according to your camera's resolution
frame_height = 720
middle_y = frame_height // 2

counter = solutions.ObjectCounter(
    view_img=False,
    reg_pts=[(0, middle_y), (frame_width, middle_y)],
    names=model.names,
    draw_tracks=True,
    line_thickness=2
)


def encode_frame(frame):
    """Encode the frame as base64 for JSON serialization."""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')


def list_cameras():
    """List available camera devices and return their indices."""
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr


def select_primary_camera(available_cameras):
    """Select the primary camera (usually the built-in webcam)."""
    for camera_index in available_cameras:
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            # Assuming the primary camera has a lower resolution than an iPhone camera
            if cap.get(cv2.CAP_PROP_FRAME_WIDTH) <= 1280 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) <= 720:
                cap.release()
                return camera_index
            cap.release()
    return available_cameras[0]  # Fallback to the first available camera


def process_video():
    """Process video feed and emit frames to client."""
    available_cameras = list_cameras()
    if len(available_cameras) == 0:
        print("No cameras found")
        return

    # Select the primary camera
    camera_index = select_primary_camera(available_cameras)
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Error: Could not open video feed")
        return

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Could not read frame")
            break

        # Process the frame
        processed_frame = model.track(
            frame, persist=True, show=False, classes=classes_to_count)
        processed_frame = counter.start_counting(frame, processed_frame)

        # Encode frames for transmission
        original_encoded = encode_frame(frame)
        processed_encoded = encode_frame(processed_frame)

        # Emit frames to the client
        socketio.emit('video_frame', {
                      'original': original_encoded, 'processed': processed_encoded})

    cap.release()


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('connect')
def handle_connect():
    socketio.start_background_task(target=process_video)


if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0')