import streamlit as st
import cv2
from camera_tampering_detection import argsParser, run_combined_detection

def main():
    st.title("Camera Tampering Detection")

    # File upload section
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"])

    if uploaded_file is not None:
        file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
        st.write(file_details)

        if st.button("Start Detection"):
            args = argsParser()
            # Save the uploaded video file
            with open("uploaded_video.mp4", "wb") as f:
                f.write(uploaded_file.getbuffer())

            video_capture = cv2.VideoCapture("uploaded_video.mp4")
            _, frame = video_capture.read()
            H, W = frame.shape[:2]
            run_combined_detection(args, W, H, video_capture)
    else:
        st.write("Upload a video file to start detection.")

if __name__ == "__main__":
    main()

