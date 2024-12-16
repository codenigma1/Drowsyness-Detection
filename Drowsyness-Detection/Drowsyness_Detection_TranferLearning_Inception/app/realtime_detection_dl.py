

import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from pygame import mixer
import time
from PIL import Image
import os


    
# Initialize the app
st.title("Drowsiness Detection System 🛑😴")
st.write("This application uses a pre-trained model to detect drowsiness in real time. 🚗💤")
st.write("### How to Use:")
st.write("1. 🎛️ Adjust the sliders to set the **Drowsiness Threshold** and **Cool-down Duration**.")
st.write("2. 🛑 Close the app or refresh the page to stop the detection.")
st.write("**⚠️ Warning**: If the app stops unexpectedly, refresh the page to restart.")


# Sidebar for app configuration
st.sidebar.title("Settings ⚙️")
drowsy_threshold = st.sidebar.slider("Drowsiness Threshold", min_value=5, max_value=30, value=15, step=1)
cool_down_duration = st.sidebar.slider("Cool-down Duration (seconds)", min_value=1, max_value=10, value=5, step=1)

# Load Haar cascades and the model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
model = tf.keras.models.load_model(os.path.join('..', 'save_models', 'new_best_model.h5'))

# Initialize the alarm sound
mixer.init()
sound = mixer.Sound(os.path.join('..', 'sounds', 'alarm2.wav'))

# Drowsiness variables
score = 0
is_playing = False  # To track sound state
cool_down_timer = 0  # Time tracking variable

# Streamlit placeholders
FRAME_WINDOW = st.image([])  # Placeholder for displaying frames
cap = None  # Initialize capture variable

# Utility functions
def initialize_video_capture():
    """Initialize the video capture and return the capture object."""
    return cv2.VideoCapture(0)

def release_video_capture(capture):
    """Safely release the video capture object."""
    if capture is not None:
        capture.release()
        cv2.destroyAllWindows()

def draw_header_bar(frame, text="Drowsiness Detection"):
    height, width = frame.shape[:2]
    bar_height = 50
    cv2.rectangle(frame, (0, 0), (width, bar_height), (50, 50, 50), -1)
    cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def draw_status_overlay(frame, status_text, color=(255, 255, 255)):
    height, width = frame.shape[:2]
    overlay_height = 60
    sub_img = frame[height - overlay_height:height, 0:width]
    overlay = sub_img.copy()
    cv2.rectangle(overlay, (0, 0), (width, overlay_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, sub_img, 0.4, 0, sub_img)
    frame[height - overlay_height:height, 0:width] = sub_img
    cv2.putText(frame, status_text, (10, height - 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)

def draw_alert_overlay(frame):
    height, width = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (width, 50), (0, 0, 255), -1)
    cv2.putText(frame, "ALERT! You are Drowsy!", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

def draw_drowsiness_bar(frame, score, max_score=30):
    height, width = frame.shape[:2]
    bar_x = width - 30
    bar_y_start = 60
    bar_height = 200
    bar_thickness = 20
    
    cv2.rectangle(frame, (bar_x, bar_y_start), (bar_x + bar_thickness, bar_y_start + bar_height), (50, 50, 50), -1)
    filled_height = int((score / max_score) * bar_height)
    filled_color = (0, 0, 255) if score > drowsy_threshold else (0, 255, 0)
    cv2.rectangle(frame, (bar_x, bar_y_start + (bar_height - filled_height)), 
                  (bar_x + bar_thickness, bar_y_start + bar_height), filled_color, -1)
    cv2.putText(frame, "Score", (bar_x - 60, bar_y_start + 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
    cv2.putText(frame, str(score), (bar_x - 40, bar_y_start + bar_height + 20), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)

# Session state for video capture
if 'run_detection' not in st.session_state:
    st.session_state.run_detection = True  # Start detection by default
if 'cap' not in st.session_state:
    st.session_state.cap = initialize_video_capture()  # Initialize capture as soon as the app loads

# Real-time detection loop
if st.session_state.run_detection:
    while True:
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.warning("Failed to initialize camera. Please check your camera.")
            break

        ret, frame = st.session_state.cap.read()
        if not ret:
            st.warning("Failed to grab frame. Please check your camera.")
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces and eyes
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)
        eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3)

        # Draw UI elements
        draw_header_bar(frame)
        draw_drowsiness_bar(frame, score)

        pred_text = ""

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color=(255, 0, 0), thickness=2)

        if len(eyes) > 0:
            (ex, ey, ew, eh) = eyes[0]
            eye_roi = frame[ey:ey + eh, ex:ex + ew]
            eye_roi = cv2.resize(eye_roi, (80, 80))
            eye_roi = eye_roi / 255.0
            eye_roi = eye_roi.reshape(1, 80, 80, 3)

            # Predict using the model
            prediction = model.predict(eye_roi, verbose=0)
            closed_prob = np.round(float(prediction[0][0]), 4)
            open_prob = np.round(float(prediction[0][1]), 4)

            pred_text = f"Closed: {closed_prob:.2f}, Open: {open_prob:.2f}"
            draw_status_overlay(frame, pred_text, (255, 255, 255))

            if closed_prob > 0.30:
                score += 1
                draw_status_overlay(frame, 'Status: DROWSY!', (0, 0, 255))
                if score > drowsy_threshold:
                    draw_alert_overlay(frame)
                    if not is_playing and time.time() - cool_down_timer > cool_down_duration:
                        sound.play(loops=-1)
                        is_playing = True
            elif open_prob > 0.90:
                score = max(score - 1, 0)
                draw_status_overlay(frame, 'Status: AWAKE', (0, 255, 0))
                if is_playing:
                    sound.stop()
                    is_playing = False
                    cool_down_timer = time.time()
            else:
                draw_status_overlay(frame, "Status: Uncertain", (255, 255, 0))
        else:
            draw_status_overlay(frame, "No eyes detected", (0, 255, 255))

        # Streamlit display
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

release_video_capture(st.session_state.cap)
st.write("Detection stopped 👋")



