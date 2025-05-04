#python -m streamlit run Main.py

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from ultralytics import YOLO
import cv2
import secrets
import cv2
import tempfile
from PIL import Image


# Set the page configuration
st.set_page_config(
    page_title="Safety Hazard Detector- Home",
    page_icon="🔍",
    layout="wide",
)

# Header
st.title("Safety Detection App")
st.markdown("### Ensuring safety through advanced detection technologies")

# Main image
st.image("App_Image.jpg", use_container_width=True)

# Introduction
st.markdown("""
Welcome to the **Safety Detection App**. 
Our app utilizes cutting-edge technology to monitor and enhance safety in various environments. 
Whether it's for home security, workplace safety, or public area monitoring, we've got you covered!
""")

# Features section
st.markdown("## Features")
features = [
    "🔒 **Real-time Alerts:** Receive instant notifications on safety breaches.",
    "🤖 **AI Detection:** Advanced algorithms for detecting threats and suspicious activities.",
    "📊 **Data Analytics:** Insights and reports to help improve safety measures.",
    "📱 **User-Friendly Interface:** Easy to use with seamless navigation.",
    "🌐 **Multi-device Support:** Access the app from your smartphone, tablet, or desktop."
]

for feature in features:
    st.markdown(f"- {feature}")

# Call to action
st.markdown("""
## Get Started Now!
To ensure your safety, click on the next pages and take the first step towards a safer environment.
""")


# Footer
st.markdown("---")
st.markdown("© 2024 Safety Detection App. All Rights Reserved.")