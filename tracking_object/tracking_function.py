import cv2
from ultralytics import YOLO
import numpy as np
import torch
import time

def open_camera():
    capture = cv2.VideoCapture(0)
    return capture

def object_tracking(capture):
    # load a pre trained model
    model = YOLO("yolo11n.pt")
    #print(torch.backends.mps.is_available())

    # check if no other detection happened in the 0.5sec time period
    last_detection_time = 0

    ret, frame = capture.read()

    frame = cv2.flip(frame, 1)
    if not ret:
        print("unable to open")
        return exit(1)

    #32 sports ball and 67 for cell phone classes
    results = model(frame, agnostic_nms = True, stream = True, classes = [32,67])

    annotated_frame = frame.copy()

    #getting current time
    current_time = time.time()

    # can't detect a object so, paddles do not move
    top = (None,None)
    top2 = (None, None)

    #detect only once per second
    if current_time - last_detection_time >= 0.5:
        results = model(frame, agnostic_nms=True, stream=True, classes=[32, 67])
        for result in results:
            annotated_frame = result.plot()
            for box in result.boxes:
                class_index = int(box.cls)
                class_name = result.names[class_index]

                x, y, w, h = box.xywh[0].cpu().numpy()
                if class_name == 'cell phone':
                    #use the top of the boundBox to control the paddle
                    top = (int(x + w / 2), int(y))
                if class_name == 'sports ball':
                    top2 = (int(x + w / 2), int(y))

            cv2.imshow("YOLO Inference", annotated_frame)

        # update the last detection time
        last_detection_time = current_time

    return frame, top, top2