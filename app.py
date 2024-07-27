import streamlit as st
import cv2
import numpy as np
import random
from ultralytics import YOLO

# Function to load class list from coco.txt
def load_class_list(file_path):
    with open(file_path, "r") as f:
        class_list = f.read().strip().split("\n")
    return class_list

# Function to generate random colors
def generate_detection_colors(class_list):
    detection_colors = []
    for _ in range(len(class_list)):
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        detection_colors.append((b, g, r))
    return detection_colors

# Load class list
class_list = load_class_list("D:/Coding/NIT FSDS/1 NIT NOTES/04 JUNE NIT/18  06 14/yolo/utils/coco.txt")

# Generate colors for each class
detection_colors = generate_detection_colors(class_list)

# Load YOLOv8 model
model = YOLO("weights/yolov8n.pt")

# Streamlit title
st.title("YOLO Object Detection")

# Streamlit file uploader
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video is not None:
    # Save the uploaded video to a file
    with open("uploaded_video.mp4", "wb") as f:
        f.write(uploaded_video.getbuffer())

    # Open the video file
    cap = cv2.VideoCapture("uploaded_video.mp4")

    if not cap.isOpened():
        st.error("Cannot open video file")
    else:
        stframe = st.empty()  # Placeholder for video frames

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Predict on the frame
            detect_params = model.predict(source=[frame], conf=0.45)

            # Convert tensor array to numpy
            DP = detect_params[0].numpy()

            if len(DP) != 0:
                for i in range(len(detect_params[0])):
                    boxes = detect_params[0].boxes
                    box = boxes[i]
                    clsID = box.cls.numpy()[0]
                    conf = box.conf.numpy()[0]
                    bb = box.xyxy.numpy()[0]

                    cv2.rectangle(
                        frame,
                        (int(bb[0]), int(bb[1])),
                        (int(bb[2]), int(bb[3])),
                        detection_colors[int(clsID)],
                        3,
                    )

                    # Display class name and confidence
                    font = cv2.FONT_HERSHEY_COMPLEX
                    cv2.putText(
                        frame,
                        class_list[int(clsID)] + " " + str(round(conf, 3)) + "%",
                        (int(bb[0]), int(bb[1]) - 10),
                        font,
                        1,
                        (255, 255, 255),
                        2,
                    )

            # Display the resulting frame
            stframe.image(frame, channels="BGR")

        cap.release()
