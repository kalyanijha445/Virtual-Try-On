from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import os
import json
import subprocess

app = Flask(__name__)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)

# Set up video capture
camera = cv2.VideoCapture(0)

@app.route('/')
def index():
    """Render the main HTML page."""
    return render_template('index.html')

def generate_frames():
    """Capture frames from the camera and process landmarks."""
    while True:
        success, frame = camera.read()
        if not success:
            break

        # Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        # Draw landmarks on the frame
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # Encode the frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    """Provide the video feed to the web page."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture', methods=['POST'])
def capture_landmarks():
    """Capture landmarks and save the frame."""
    success, frame = camera.read()
    if not success:
        return jsonify({"error": "Failed to capture frame."}), 500

    # Convert frame to RGB for MediaPipe processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if not results.pose_landmarks:
        return jsonify({"error": "No landmarks detected."}), 400

    # Save landmarks to JSON
    landmarks_data = []
    for landmark in results.pose_landmarks.landmark:
        landmarks_data.append({
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z
        })

    landmarks_file = "landmarks.json"
    with open(landmarks_file, "w") as f:
        json.dump(landmarks_data, f)

    # Save the captured frame
    output_image = "captured_frame.jpg"
    cv2.imwrite(output_image, frame)

    return jsonify({"message": "Frame and landmarks captured successfully."})

@app.route('/generate_model_page')
def generate_model_page():
    """Render the model generation page."""
    # Trigger the model generation script
    subprocess.run(['python', 'generate_model_with_shirt.py'], check=True)

    # Render the page with shirt options
    shirts = ["shirt1.png", "shirt2.png", "shirt3.png", "shirt4.png", "shirt5.png"]
    return render_template('generate_model.html', shirts=shirts)

@app.route('/apply_shirt', methods=['POST'])
def apply_shirt():
    """Apply the selected shirt to the model."""
    selected_shirt = request.json.get('shirt')
    if not selected_shirt:
        return jsonify({"error": "No shirt selected."}), 400

    # Logic to overlay shirt on model can go here
    return jsonify({"message": f"Shirt {selected_shirt} applied to model!"})

if __name__ == '__main__':
    app.run(debug=True)
