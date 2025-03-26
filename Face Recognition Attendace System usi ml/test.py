

import cv2
import pickle
import numpy as np
import csv
import time
from datetime import datetime
import os
from sklearn.neighbors import KNeighborsClassifier
from win32com.client import Dispatch  # Import the voice module

# Function for speech synthesis
def speak(message):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(message)

# Load the faces and labels from the pickle file
with open('data/faces_and_labels.pkl', 'rb') as f:
    data = pickle.load(f)

faces_data = data['faces']
labels = data['labels']

# Train a KNN classifier on the face data and labels
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces_data, labels)

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")

# Initialize attendance list
attendance_list = []

# CSV File for storing attendance
attendance_file = "attendance.csv"
if not os.path.isfile(attendance_file):
    # Create the CSV file and write headers if it doesn't exist
    with open(attendance_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Name", "Timestamp"])

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resize_img = cv2.resize(crop_img, (50, 50))  # Resize to match the training data
        flatten_img = resize_img.flatten().reshape(1, -1)  # Flatten and reshape for prediction
        
        # Predict the label of the face
        predicted_label = knn.predict(flatten_img)[0]
        
        # Draw bounding box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y-40), (x + w, y), (0, 0, 255), -1)
        cv2.putText(frame, predicted_label, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    
    if k == ord('o'):  # If 'o' is pressed, take attendance
        ts = time.time()
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%y")
        
        # Record attendance in the list and the CSV
        attendance_list.append({"name": predicted_label, "time": timestamp})
        
        # Store the attendance in the CSV file
        with open(attendance_file, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([predicted_label, timestamp])
        
        # Announce attendance
        speak(f"Attendance is taken for {predicted_label} at {timestamp}")
        print(f"Attendance taken for {predicted_label} at {timestamp}")
    
    if k == ord('q'):  # Exit the loop when 'q' is pressed
        break

video.release()
cv2.destroyAllWindows()
