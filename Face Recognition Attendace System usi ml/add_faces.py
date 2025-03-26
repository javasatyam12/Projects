

import cv2
import pickle
import numpy as np
import os

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
faceDetect = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
faces_data = []  # List to hold face data
labels = []  # List to hold corresponding labels

i = 0
name = input("Enter your name: ")

# Collect faces data
while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resize_img = cv2.resize(crop_img, (50, 50))  # Resize to ensure the same size
        
        if len(faces_data) < 100 and i % 10 == 0:  # Collect up to 100 faces, spaced by 10 frames
            faces_data.append(resize_img.flatten())  # Flatten the face image
            labels.append(name)  # Append the label (name) for each face
        
        i += 1
        
        # Display the number of samples collected
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:  # Exit condition
        break

video.release()
cv2.destroyAllWindows()

# Convert faces data into numpy array
faces_data = np.array(faces_data)  # Shape (100, 2500) if resized to 50x50

# Combine faces_data and labels into a dictionary to save together
data = {'faces': faces_data, 'labels': labels}

# Save the data as a new pickle file
with open('data/faces_and_labels.pkl', 'wb') as f:
    pickle.dump(data, f)

print(f"Collected {len(faces_data)} faces data for {name} and saved to faces_and_labels.pkl")