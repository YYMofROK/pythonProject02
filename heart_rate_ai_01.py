import cv2
import time
import numpy as np

from utils.inference import draw_bounding_box

# Define constants
WINDOW_NAME = 'Heart Rate Monitor'
FACE_CASCADE_FILE = './models/haarcascade_frontalface_default.xml'

# Load face detection model
face_cascade = cv2.CascadeClassifier(FACE_CASCADE_FILE)

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define heart rate calculation variables
heart_rates = []
heart_rate_history = []
heart_rate_average = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Flip the frame horizontally
    mirror_image = cv2.flip(frame, 1)

    # Convert to grayscale
    gray_image = cv2.cvtColor(mirror_image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the cascade classifier
    faces = face_cascade.detectMultiScale(
        gray_image
        , scaleFactor=1.1
        , minNeighbors=5
        , flags=cv2.CASCADE_SCALE_IMAGE
        , minSize=(250, 250)
        , maxSize=(500, 500)
    )

    # Process detected faces
    for face_coordinates in faces:
        x, y, w, h = face_coordinates

        # Extract the facial region
        facial_region = frame[y:y + h, x:x + w]

        # Calculate the average red color value in the facial region
        average_red_value = cv2.mean(facial_region, mask=None)[0]

        # Append the average red value to the heart rate history
        heart_rates.append(average_red_value)

        # Calculate the average heart rate based on the last 10 readings
        if len(heart_rates) >= 10:
            heart_rate_history = heart_rates[-10:]
            heart_rate_average = np.mean(heart_rate_history)
            heart_rate_average = heart_rate_average * 0.7       # RGB - R add value

        # Draw a filled bounding box around the detected face
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), cv2.FILLED)

        # Display the current heart rate
        cv2.putText(mirror_image, f"Heart Rate: {heart_rate_average:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw a bounding box around the detected face
        draw_bounding_box(face_coordinates, mirror_image, (0, 255, 0) )

    # Display the captured frame
    cv2.imshow(WINDOW_NAME, mirror_image)

    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
