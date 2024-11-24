import cv2
import mediapipe as mp
import json
import os

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False)

# Open Camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Press 'c' to capture the frame with landmarks or 'q' to quit.")
landmarks_data = []

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    # Convert the image to RGB for MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    # Draw landmarks if detected
    annotated_frame = frame.copy()
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

    # Display video feed
    cv2.imshow("Body Scanning", annotated_frame)

    # Capture key input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):  # Capture landmarks on pressing 'c'
        if results.pose_landmarks:
            print("Capturing landmarks and saving image...")
            for landmark in results.pose_landmarks.landmark:
                landmarks_data.append({
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })

            # Save landmarks to a JSON file
            landmarks_file = "landmarks.json"
            with open(landmarks_file, "w") as f:
                json.dump(landmarks_data, f)
            print(f"Landmarks saved to {landmarks_file}")

            # Save captured frame as an image
            output_image = "captured_frame.jpg"
            cv2.imwrite(output_image, frame)
            print(f"Captured image saved as {output_image}")
        else:
            print("No landmarks detected. Try again.")
        break

    elif key == ord('q'):  # Quit
        print("Quitting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
