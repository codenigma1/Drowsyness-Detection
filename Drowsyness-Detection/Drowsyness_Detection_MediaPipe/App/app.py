
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from pygame import mixer
from utils import calculate_avg_ear, denormalize_coords  # Your utility functions
import os

#############################
# Streamlit UI
#############################

st.title("Drowsiness Detection with Mediapipe & Streamlit")

# We’ll store the app’s running state in st.session_state.
if 'is_running' not in st.session_state:
    st.session_state.is_running = False

# Sidebar or top-level buttons
col1, col2 = st.columns(2)
with col1:
    start_button = st.button("Start Detection", key="start_button")
with col2:
    stop_button = st.button("Stop Detection", key="stop_button")

# Update running state
if start_button:
    st.session_state.is_running = True
if stop_button:
    st.session_state.is_running = False

# Container to display the video feed
video_container = st.empty()

#############################
# Constants & Setup
#############################

EAR_THRESHOLD = 0.25
CONSECUTIVE_FRAMES = 15

# Landmarks (from your original code)
chosen_left_eye_idxs = [362, 385, 387, 263, 373, 380]
chosen_right_eye_idxs = [33, 160, 158, 133, 153, 144]
all_chosen_eye_idxs = chosen_left_eye_idxs + chosen_right_eye_idxs

frame_count = 0
status = "AWAKE"
alert_triggered = False

# Initialize mixer outside the loop
mixer.init()
# alarm_sound = mixer.Sound('Drowsyness_Detection_MediaPipe/media/alarm.mp3')
alarm_sound = mixer.Sound(os.path.join('..', 'media', 'alarm.mp3'))

# Mediapipe face mesh
mp_facemesh = mp.solutions.face_mesh

#############################
# Main Detection Function
#############################

def run_drowsiness_detection():
    global frame_count, status, alert_triggered

    # Start video capture
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # sometimes CAP_DSHOW helps on Windows

    # Using "with" statement for FaceMesh context manager
    with mp_facemesh.FaceMesh(
        refine_landmarks=True, 
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5
    ) as face_mesh:
        
        while True:
            # Check if user requested to stop
            if not st.session_state.is_running:
                break

            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera. Please check your camera device.")
                break

            imgH, imgW, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame for face landmarks
            results = face_mesh.process(frame_rgb)
            color = (0, 255, 0)  # Default to green

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Calculate EAR
                    landmarks = face_landmarks.landmark
                    avg_ear, _ = calculate_avg_ear(
                        landmarks, 
                        chosen_left_eye_idxs, 
                        chosen_right_eye_idxs, 
                        imgW, 
                        imgH
                    )

                    # Update status based on EAR
                    if avg_ear < EAR_THRESHOLD:
                        frame_count += 1
                        if frame_count >= CONSECUTIVE_FRAMES and not alert_triggered:
                            status = "DROWSY"
                            alert_triggered = True
                            color = (0, 0, 255)  # Red
                            alarm_sound.play()
                    else:
                        frame_count = 0
                        status = "AWAKE"
                        alert_triggered = False
                        color = (0, 255, 0)  # Green
                        alarm_sound.stop()

                    # Overlay EAR and status text
                    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Status: {status}", (30, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                    # Display alert banner
                    if status == "DROWSY":
                        cv2.rectangle(frame, (0, imgH // 2 - 50), 
                                      (imgW, imgH // 2 + 50), 
                                      (0, 0, 255), -1)
                        cv2.putText(frame, "ALERT: DROWSY!", 
                                    (imgW // 4, imgH // 2 + 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, 
                                    (255, 255, 255), 3)

                    # Draw eye landmarks
                    for idx in all_chosen_eye_idxs:
                        landmark = landmarks[idx]
                        coord = denormalize_coords(landmark.x, landmark.y, imgW, imgH)
                        if coord:
                            cv2.circle(frame, coord, 2, color, -1)

            # Convert BGR to RGB for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_container.image(frame_rgb, channels='RGB', use_container_width=True)

    # Release resources
    cap.release()
    alarm_sound.stop()

#############################
# Run the detection if started
#############################

if st.session_state.is_running:
    run_drowsiness_detection()
else:
    st.info("Click **Start Detection** to begin the real-time drowsiness detection.")
