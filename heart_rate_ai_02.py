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

# Define variables to track reference face color
reference_face_color = np.array([0, 0, 0, 1])

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert to grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the cascade classifier
    faces = face_cascade.detectMultiScale(
        gray_image, scaleFactor=1.5, minNeighbors=5, minSize=(250, 250), flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Process detected faces
    for face_coordinates in faces:
        x, y, w, h = face_coordinates

        # Extract the entire facial region
        facial_region = frame[y:y + h, x:x + w]

        # Calculate the average RGB color values in the entire facial region
        average_color = cv2.mean(facial_region)

        # Calculate the overall color change from the reference face color
        if np.linalg.norm(reference_face_color - average_color) > 0:
            color_change = reference_face_color - average_color
            color_change_intensity = np.linalg.norm(color_change)
        else:
            color_change = np.array([0, 0, 0, 1])
            color_change_intensity = 0

        # Append the color change intensity to the heart rate history, regardless of its sign
        heart_rates.append(color_change_intensity)

        # Calculate the average heart rate based on the last 10 readings
        if len(heart_rates) >= 10:
            heart_rate_history = heart_rates[-10:]
            heart_rate_average = np.mean(heart_rate_history)
            heart_rate_average = heart_rate_average * 0.3

        # Display the current heart rate
        cv2.putText(frame, f"Heart Rate: {heart_rate_average:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Draw a bounding box around the detected face
        draw_bounding_box(face_coordinates, frame, (0, 255, 0))

    # Display the captured frame
    cv2.imshow(WINDOW_NAME, frame)

    # Check for 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
