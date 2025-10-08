import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Streamlit page setup
st.set_page_config(page_title="EduPulse - AI Learning Energy Tracker", layout="wide")
st.title("🧠 EduPulse – AI Learning Energy Tracker")
st.markdown(
    "Monitor your **study focus**, get **AI-based recommendations**, and build better learning habits!"
)

# Initialize session state variables
if "focus_log" not in st.session_state:
    st.session_state.focus_log = []
if "focused_time" not in st.session_state:
    st.session_state.focused_time = 0
if "distracted_time" not in st.session_state:
    st.session_state.distracted_time = 0
if "frame_count" not in st.session_state:
    st.session_state.frame_count = 0

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Camera input
uploaded_file = st.camera_input("📸 Please stay in front of your camera")

if uploaded_file:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Determine focus state
    focus_state = "Focused" if len(faces) > 0 else "Distracted"
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update counters
    st.session_state.frame_count += 1
    if focus_state == "Focused":
        st.session_state.focused_time += 1
    else:
        st.session_state.distracted_time += 1

    # Log every 5 frames
    if st.session_state.frame_count % 5 == 0:
        timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.focus_log.append([timestamp, focus_state])

    # Display frame
    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption=f"Status: {focus_state}", use_column_width=True)

    # Show focus summary
    total_time = st.session_state.focused_time + st.session_state.distracted_time
    focus_ratio = (st.session_state.focused_time / total_time) * 100 if total_time > 0 else 0
    df = pd.DataFrame(st.session_state.focus_log, columns=["Time", "State"])

    st.subheader("📊 Focus Summary")
    st.write(df)

    st.metric(label="Total Frames Captured", value=total_time)
    st.metric(label="Focus Percentage", value=f"{focus_ratio:.1f}%")

    # AI-like recommendations
    st.subheader("🤖 Smart Study Suggestions")
    if focus_ratio >= 85:
        st.success("Amazing focus! You’re in the zone. Take a short 5-minute relaxation break.")
    elif 60 <= focus_ratio < 85:
        st.info("Good job! Try to reduce distractions. Maybe silence your phone and refocus.")
    else:
        st.warning("You seem distracted. Take a 10-minute walk or stretch. Try again after refreshing yourself.")

    # Download focus log
    csv = df.to_csv(index=False)
    st.download_button("Download Focus Log (CSV)", csv, "focus_log.csv")

else:
    st.info("Click on the camera input above to begin tracking your attention in real-time.")
