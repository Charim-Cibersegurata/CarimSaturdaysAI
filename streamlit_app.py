import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2

st.title("My first Streamlit app")
st.write("Hello, world")
ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)
