#python -m streamlit run Main.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import secrets


def Main_App():
    
    st.set_page_config(page_title= "Safety Hazard Detector- Image", layout = "wide", page_icon ="⛑️")


    st.title('Safety Hazard Detector')

    st.sidebar.markdown("# Safety Hazard Detection ⛑️")

    with st.sidebar.form(key='image_upload_form'):
        st.header("Image Upload Form")

        # File uploader for image files
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        confidence = st.slider(
                "Confidence Threshold", 
                0.0, 1.0, 0.5, 
                key='img_conf')
        # Submit button
        submit_button = st.form_submit_button(label='Submit Image')

    # Process the uploaded image
    if submit_button and uploaded_file is not None:
        # Opening the image using PIL
        image = Image.open(uploaded_file)

        # Display the image
        #st.image(image, caption='Uploaded Image', use_container_width=True)

    model = YOLO("last.pt")


    # Creating two columns
    col1, col2 = st.columns(2)

    with col1:
        if uploaded_file is not None:
            image_col1 = Image.open(uploaded_file)
            st.image(image_col1, caption='Uploaded Image in Column 1', use_container_width=True)

            image_np = np.array(image_col1)

            # Convert color from RGB to BGR for OpenCV
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

            # Submit button
            if st.button("PREDICT HAZARD ⚠️") and image_bgr is not None:
                st.session_state.image = image_bgr  # Store the image in session state

                    
        

    with col2:
        
        if 'image' in st.session_state:
            prediction = model.predict(image_bgr, imgsz=640, conf=confidence)
            result_image = prediction[0].plot()
            st.image(result_image, caption='Predicted hazard', use_container_width=True)
        else:
            st.write("No image uploaded yet.")

    

Main_App()
