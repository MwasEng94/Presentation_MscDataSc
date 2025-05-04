#python -m streamlit run Main.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import secrets
import tempfile

import streamlit as st
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from PIL import Image

# Page Config
st.set_page_config(
    page_title="Safety Hazard Detector- Video 🎬",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon ="⛑️"
)

# Title
st.title("Safety Hazard Detector")
st.markdown("Upload a video and run inference in real-time!")

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.01)


model = 'last.pt'

# Upload Video
uploaded_file = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

# Load YOLOv8 Model (cached)
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)

model = load_model(model)

# Process Video Function
def process_video(video_path):
   
    cap = cv2.VideoCapture(0)  # Webcam
    
    if not cap.isOpened():
        st.error("Error opening video file!")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    st.info(f"📹 Video Info: {width}x{height} @ {fps:.2f} FPS")
    
    # Create Streamlit video placeholder
    video_placeholder = st.empty()
    
    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLOv8 inference
        results = model(frame, conf=confidence_threshold)
        
        # Draw bounding boxes
        annotated_frame = results[0].plot()
        
        # Display in Streamlit
        video_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
    
    cap.release()

# Main App Logic
if uploaded_file:
    # Save uploaded video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    
    # Process video
    with st.spinner("🔍 Detecting objects..."):
        process_video(tfile.name)
    
    # Clean up
    tfile.close()
else:
    st.warning("⚠️ Please upload a video file to begin detection.")

st.markdown("---")
st.markdown("**💡 Tip:** Use the sidebar to adjust confidence threshold.")
