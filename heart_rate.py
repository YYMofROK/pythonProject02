import cv2
import time
import numpy as np
from keras.models import load_model
from statistics import mode
from utils.datasets import get_labels
from utils.inference import detect_faces
from utils.inference import draw_text
from utils.inference import draw_bounding_box
from utils.inference import apply_offsets
from utils.inference import load_detection_model
from utils.preprocessor import preprocess_input

# loading models
face_cascade = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')

USE_WEBCAM = True  # If false, loads video file source
# USE_WEBCAM = False # If false, loads video file source

# starting video streaming
cv2.namedWindow('window_frame')
video_capture = cv2.VideoCapture(0)

# Select video or webcam feed
cap = None
if (USE_WEBCAM == True):
    cap = cv2.VideoCapture(0)  # Webcam source
    # 현재 카메라 해상도 얻기
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print('default resolution width {} height {}'.format(width, height))

    # 해상도 변경
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 400)
    # print('default resolution width {} height {}'.format(width, height))

else:
    cap = cv2.VideoCapture('./demo/dinner.mp4')  # Video file source

increment = 0
while cap.isOpened():  # True:
    ret, bgr_image = cap.read()
    bgr_image = cv2.flip(bgr_image, 1)
    # bgr_image = video_capture.read()[1]

    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    faces = face_cascade.detectMultiScale(
        gray_image
        , scaleFactor=1.1
        , minNeighbors=5
        , minSize=(250, 250)
        , flags=cv2.CASCADE_SCALE_IMAGE
    )

    for face_coordinates in faces:
        color = np.asarray((0, 255, 0))
        color = color.astype(int)
        color = color.tolist()
        draw_bounding_box(face_coordinates, rgb_image, color)
        #print(face_coordinates)
        x, y, w, h = face_coordinates
        cropped_face = rgb_image[y:y + h, x:x + w]
        cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
        increment = increment + 1
        cv2.imwrite('./img_face/only_face_' + str(increment) + '.jpg', cropped_face)

    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('window_frame', bgr_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
