import cv2
import math
from ultralytics import YOLO
from gtts import gTTS
import pygame
import tempfile
import os

# Load the YOLO model
model = YOLO("yolo-Weights/yolov8n.pt")

# Load the object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Define a function to convert object text to audible format
def play_audio(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    temp_dir = tempfile.gettempdir()
    temp_audio_path = os.path.join(temp_dir, "temp_audio.mp3")
    try:
        tts.save(temp_audio_path)
        pygame.mixer.init()
        pygame.mixer.music.load(temp_audio_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    except PermissionError:
        print(f"PermissionError: Failed to save or load audio file in {temp_dir}")
    finally:
        try:
            os.remove(temp_audio_path)
        except Exception as e:
            print(f"Failed to remove temporary file: {e}")

# Calculate the focal length using the known width and distance
def calculate_focal_length(known_distance, known_width, pixel_width):
    return (pixel_width * known_distance) / known_width

# Define a function to calculate the distance between the object and the camera
def calculate_distance(focal_length, known_width, pixel_width):
    return (known_width * focal_length) / pixel_width

# Start the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Calibration values (example: known object width = 0.5 meters, known distance = 2.0 meters)
known_width = 0.5  # meters
known_distance = 2.0  # meters
calibrated = False
focal_length = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Draw bounding boxes and print object details
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            confidence = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

            # Convert object text to audible format
            object_text = f"Object detected: {classNames[cls]}"
            play_audio(object_text)

            # Calculate the focal length if not calibrated
            pixel_width = x2 - x1
            if not calibrated:
                focal_length = calculate_focal_length(known_distance, known_width, pixel_width)
                calibrated = True

            # Calculate the distance between the object and the camera
            obj_distance = calculate_distance(focal_length, known_width, pixel_width)
            distance_text = f"The distance to the object is approximately {obj_distance:.2f} meters."
            print(distance_text)
            play_audio(distance_text)

            # Alert if the object is too near
            if obj_distance < 0.5:  # adjust the threshold as needed
                alert_text = "Alert: Object is too near to the camera!"
                print(alert_text)
                play_audio(alert_text)

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
