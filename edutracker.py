import streamlit as st
import cv2
import mediapipe as mp
import time
import pandas as pd
import numpy as np
from datetime import datetime

# Initialize mediapipe face detection
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# Streamlit UI setup
st.set_page_config(page_title="EduPulse - AI Learning Energy Tracker", layout="wide")
st.title("🧠 EduPulse – AI Learning Energy Tracker")
st.markdown("Monitor your **study focus**, get **AI-based recommendations**, and build better learning habits!")

# Initialize variables
focus_log = []
focus_start = None
focused_time = 0
distracted_time = 0
frame_count = 0
focus_state = "Not started"

start_button = st.button("Start Focus Session")

if start_button:
    st.write("📸 Webcam starting... Please stay in front of your camera.")
    stframe = st.empty()
    face_detection = mp_face.FaceDetection(min_detection_confidence=0.6)
    cap = cv2.VideoCapture(0)

    session_start = time.time()
    focus_start = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("No camera feed detected. Please check your webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb)

        if results.detections:
            focus_state = "Focused"
            mp_draw.draw_detection(frame, results.detections[0])
        else:
            focus_state = "Distracted"

        # Time tracking
        frame_count += 1
        if focus_state == "Focused":
            focused_time += 1
        else:
            distracted_time += 1

        # Update dashboard
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Log focus data every 5 seconds
        if frame_count % 150 == 0:  # approx 5 sec
            timestamp = datetime.now().strftime("%H:%M:%S")
            focus_log.append([timestamp, focus_state])
            st.write(f"🕒 {timestamp}: {focus_state}")

        # Stop after 3 minutes (for demo)
        if time.time() - session_start > 180:
            st.success("Session complete! Great job.")
            break

    cap.release()

    # Create focus summary
    total_time = focused_time + distracted_time
    focus_ratio = (focused_time / total_time) * 100 if total_time > 0 else 0
    df = pd.DataFrame(focus_log, columns=["Time", "State"])

    st.subheader("📊 Focus Summary")
    st.write(df)

    st.metric(label="Total Time (seconds)", value=total_time)
    st.metric(label="Focus Percentage", value=f"{focus_ratio:.1f}%")

    # AI-like recommendation logic
    st.subheader("🤖 Smart Study Suggestions")
    if focus_ratio >= 85:
        st.success("Amazing focus! You’re in the zone. Take a short 5-minute relaxation break.")
    elif 60 <= focus_ratio < 85:
        st.info("Good job! Try to reduce distractions. Maybe silence your phone and refocus.")
    else:
        st.warning("You seem distracted. Take a 10-minute walk or stretch. Try again after refreshing yourself.")

    # Save session data
    df.to_csv("focus_log.csv", index=False)
    st.download_button("Download Focus Log (CSV)", df.to_csv(index=False), "focus_log.csv")

else:
    st.info("Click **Start Focus Session** to begin tracking your attention in real-time.")
