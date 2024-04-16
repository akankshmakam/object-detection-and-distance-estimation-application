import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import cv2
import cvzone
import math
import time

root = tk.Tk()

# window size
width = 1920
height = 1080
root.geometry(f"{width}x{height}")

# guide text
guide_text = """
User Guide

Welcome to the Object Detection and Distance Estimator Application!

This application allows you to perform object detection using your webcam or uploaded images. You can also estimate the distance between the webcam and certain objects. Follow the 
instructions below to use the app effectively.

1. Start Webcam

Click on the "Start Webcam" button to activate your webcam. This will enable real-time object detection using your computer's camera. Make sure your webcam is connected and functional.

2. Stop Webcam

To stop the webcam feed, click on the "Stop Webcam" button. This will pause the real-time object detection and freeze the last captured frame. You can resume the webcam feed by clicking 
on the "Start Webcam" button again.

3. Guide for Distance Estimation

This application can estimate the distance between the webcam and specific objects. When an object is detected, the estimated distance will be displayed on the screen. Please note that 
distance estimation is available for a limited number of objects. Refer to the list of supported objects below:

- Cell Phone
- Keyboard
- Bottle
- Book

Important Notes:

- Ensure proper lighting conditions for better object detection accuracy.
- Maintain a clear view of the objects you want to detect, avoiding occlusion or excessive clutter.
- The distance estimation feature is only available for specific objects as mentioned above.
- For any assistance or issues, please refer to the Help or Contact sections of the application.
- Visit this link to see which all objects the model can detect (COCO dataset): https://cocodataset.org

Enjoy using the Computer Vision Application for object detection and distance estimation!
"""

def display_guide():
    text_label.config(state=tk.NORMAL)  # Enable editing the text widget
    text_label.delete("1.0", tk.END)  # Clear existing content
    text_label.insert(tk.END, guide_text)  # Insert the guide text
    text_label.config(state=tk.DISABLED)  # Disable editing the text widget
    text_label.configure(font=("Arial", 12, "bold"))  # Set the font to bold

# style for buttons
style = ttk.Style()
style.configure("Custom.TButton", foreground="black", background="black", font=("Arial", 12, "bold"))

# creating all buttons
button1 = ttk.Button(root, text="Start Webcam", style="Custom.TButton")
button2 = ttk.Button(root, text="Stop Webcam", style="Custom.TButton")
button4 = ttk.Button(root, text="Guide", style="Custom.TButton", command=display_guide)

# arranging
button1.grid(row=1, column=1, padx=10, pady=10)
button2.grid(row=1, column=2, padx=10, pady=10)
button4.grid(row=1, column=4, padx=10, pady=10)

# Create a text widget for displaying guide
text_label = tk.Text(root, font=("Arial", 10), width=160, height=35, state=tk.DISABLED)
text_label.grid(row=2, column=1, columnspan=4, padx=10, pady=10)

# Computer Vision Model
cap = cv2.VideoCapture(0)

model = YOLO("../Yolo-Weights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

cell_object_width = 15
cell_d = 46
bottle_object_width = 35
bottle_d = 46
kb_object_width = 30.75
kb_d = 46
book_object_width = 14
book_d = 46

prev_frame_time = 0
new_frame_time = 0

def process_frame():
    global prev_frame_time, new_frame_time
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            class_index = int(box.cls[0])

            # Check if the detected class is a cell phone
            if classNames[class_index] == "cell phone":
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                cell_focal_length = (cell_object_width * cell_d) / w
                cell_apparent_width = w  # Apparent width of the cell phone in pixels
                cell_distance = ((cell_object_width * 3.9) / cell_apparent_width) * 100
                cvzone.putTextRect(img, f'cell phone, distance: {int(cell_distance)}cm', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1)
            elif classNames[class_index] == "bottle":
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                bottle_focal_length = (bottle_object_width * bottle_d) / w
                bottle_apparent_width = w  # Apparent width of the cell phone in pixels
                bottle_distance = ((bottle_object_width * 16.75) / bottle_apparent_width) * 9.2
                cvzone.putTextRect(img, f'bottle, distance: {int(bottle_distance)}cm', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1)
            elif classNames[class_index] == "keyboard":
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                kb_focal_length = (kb_object_width * kb_d) / w
                kb_apparent_width = w  # Apparent width of the cell phone in pixels
                kb_distance = ((kb_object_width * 2.65) / kb_apparent_width) * 291
                cvzone.putTextRect(img, f'keyboard, distance: {int(kb_distance)}cm', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1)
            elif classNames[class_index] == "book":
                cvzone.cornerRect(img, (x1, y1, w, h))
                conf = math.ceil((box.conf[0] * 100)) / 100
                book_focal_length = (book_object_width * book_d) / w
                book_apparent_width = w  # Apparent width of the cell phone in pixels
                book_distance = ((book_object_width * 1.69) / book_apparent_width) * 901
                cvzone.putTextRect(img, f'book, distance: {int(book_distance)}cm', (max(0, x1), max(35, y1)),
                                   scale=1, thickness=1)
            else:
                cls = int(box.cls[0])
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    root.after(1, process_frame)

def start_webcam():
    process_frame()

def stop_webcam():
    cap.release()
    cv2.destroyAllWindows()


def display_guide():
    text_label.config(state=tk.NORMAL)  # Enable editing the text widget
    text_label.delete("1.0", tk.END)  # Clear existing content
    text_label.insert(tk.END, guide_text)  # Insert the guide text
    text_label.config(state=tk.DISABLED)  # Disable editing the text widget
    text_label.configure(font=("Arial", 12, "bold"))  # Set the font to bold

# Button actions
button1.configure(command=start_webcam)
button2.configure(command=stop_webcam)

root.mainloop()
