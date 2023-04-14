import cv2

import numpy as np

import torch

import tensor

import keras

import pytesseract

from yolov5 import YOLOv5

from fastcnn import FastCNN

# Load the license plate detection model

model = YOLOv5()

model.load_model("yolov5s.pt")

# Load the color recognition model

model_color = FastCNN()

model_color.load_model("fastcnn_color.h5")

# Load the make and model recognition model

model_make_model = keras.models.load_model("make_model.h5")

# Initialize the CSV file

csv_file = open("data.csv", "w")

csv_writer = csv.writer(csv_file)

# Start capturing frames from the camera

cap = cv2.VideoCapture(0)

while True:

    # Capture a frame

    ret, frame = cap.read()

    # If the frame was not captured, break

    if not ret:
      
        break
        

    # Convert the frame to grayscale

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect the license plate in the frame

    plates = model.detect(gray, conf_thres=0.5)

    # If a license plate was detected, process it

    if plates:

        # Get the coordinates of the license plate

        x1, y1, x2, y2 = plates[0]

        # Crop the license plate from the frame

        plate = frame[y1:y2, x1:x2]

        # Recognize the plate number

        plate_number = pytesseract.image_to_string(plate, lang="eng")

        # Recognize the car color

        color = model_color.predict(plate)

        # Get the current time

        now = datetime.now()

        # Write the data to the CSV file

        csv_writer.writerow([now, color, plate_number])

        # Save the image of the license plate

        cv2.imwrite("plate.jpg", plate)

        # Capture the front and backside of the car

        front = frame[y1:y2, 0:x1]

        backside = frame[y1:y2, x2:]

        # Save the images of the car

        cv2.imwrite("front.jpg", front)

        cv2.imwrite("backside.jpg", backside)

        # Recognize the make and model of the car

        make, model = model_make_model.predict([front, backside])

        # Print the data

        print("License plate number:", plate_number)

        print("Car color:", color)

        print("Make:", make)

        print("Model:", model)
        # Close the CSV file

csv_file.close()

# Release the camera

cap.release()
