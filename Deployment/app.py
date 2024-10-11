# app.py

import streamlit as st
from processor import process_image, process_video, process_frame
import os
from PIL import Image
import tempfile
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, RTCConfiguration
import cv2

# Set page configuration
st.set_page_config(page_title="Traffic Violation Detection", layout="wide")

st.title("🚦 Motorbike Violation Detection App")


# Sidebar options
option = st.sidebar.radio("Select Option:", ("Image", "Video", "Live Camera"))



if option == "Image":
    st.header("🖼️ Upload and Process Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        if st.button("Process Image"):
            with st.spinner("Processing..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                    image.save(tmp.name)
                    frame = cv2.imread(tmp.name)
                processed = process_image(tmp.name, "alfont_com_arial-1.ttf")
                if processed is not None:
                    st.image(processed, caption='Processed Image.', use_column_width=True)
                    # Save processed image to temporary file for download
                    _, img_encoded = cv2.imencode('.jpg', processed)
                    st.download_button(
                        label="📥 Download Image",
                        data=img_encoded.tobytes(),
                        file_name="processed_image.jpg",
                        mime="image/jpeg"
                    )
                else:
                    st.error("Failed to process the image.")

elif option == "Video":
    st.header("🎥 Upload and Process Video")
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        # Save the uploaded video to a temporary file
        with open("temp_video.mov", "wb") as f:
            f.write(uploaded_video.read())
        
        # Display the uploaded video
        st.video("temp_video.mov")
        
        if st.button("Process Video"):
            with st.spinner("Processing Video..."):
                processed_path = process_video("temp_video.mov")
                if processed_path and os.path.exists(processed_path):
                    st.success("Video processed successfully!")
                    st.video(processed_path)
                    with open(processed_path, "rb") as file:
                        st.download_button(
                            label="📥 Download Processed Video",
                            data=file,
                            file_name="processed_video.mov",
                            mime="video/mov"
                        )
                else:
                    st.error("Failed to process the video.")


elif option == "Live Camera":
    st.header("📷 Live Camera Feed")
    st.info("Live processing is active. Detected violations will be annotated on the video feed.")

    # RTC Configuration for streamlit-webrtc
    RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    class VideoTransformer(VideoTransformerBase):
        def __init__(self):
            self.font_path = "alfont_com_arial-1.ttf"

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")  # Convert the frame to a numpy array
            processed_img = process_frame(img, self.font_path)  # Process the image using your custom function
            return processed_img

    # Start the WebRTC streaming
    webrtc_ctx = webrtc_streamer(
        key="live-camera",
        rtc_configuration=RTC_CONFIGURATION,
        video_transformer_factory=VideoTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=True
    )

    # Optional: Provide a stop button to end the live stream
    if webrtc_ctx.video_receiver:
        st.button("Stop Live Camera", on_click=webrtc_ctx.stop_stream)
