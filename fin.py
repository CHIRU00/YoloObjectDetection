import random
import cv2
import numpy as np
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
# model = YOLO('yolov8n.pt')

# Display model information (optional)
# model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data='coco8.yaml', epochs=100, imgsz=640)
cv2.namedWindow("ObjectDetection", cv2.WINDOW_NORMAL)
# Opening the file in read mode
my_file = open(r"yolo/utils/coco.txt", "r")
# Reading the file
data = my_file.read()
# Splitting the text by newline ('\n')
class_list = data.split("\n")
my_file.close()

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    detection_colors.append((b, g, r))

# Load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt")

# Vals to resize video frames | small frame optimizes the run
frame_wid = 640
frame_hyt = 480

# Open the video file
cap = cv2.VideoCapture(r"D:\Coding\NIT FSDS\1 NIT NOTES\04 JUNE NIT\18  06 14\yolo\video\854204-hd_1920_1080_30fps.mp4")

if not cap.isOpened():
    print("Cannot open video file")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # If frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize the frame | small frame optimizes the run
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # Predict on image
    detect_params = model.predict(source=[frame], conf=0.45, save=True)

    # Convert tensor array to numpy
    DP = detect_params[0].numpy()
    print(DP)

    if len(DP) != 0:
        for i in range(len(detect_params[0])):
            print(i)

            boxes = detect_params[0].boxes
            box = boxes[i]  # Returns one box
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
    cv2.imshow("ObjectDetection", frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord("q"):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
