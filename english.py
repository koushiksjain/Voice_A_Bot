import cv2
import pytesseract
from gtts import gTTS
import tempfile
import pygame
import numpy as np
from tensorflow.keras.models import load_model
import os

# Function to play audio using pygame
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

# Function to capture and process image for OCR
def ocr_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang='eng')
    print(f"OCR ENG: {text}")
    return text.strip()

# Function to recognize synthetic text using a trained model
def recognize_synthetic_text(frame, synthetic_text_model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_preprocessed = cv2.resize(gray_frame, (128, 128))  # Adjust size as per model requirements
    frame_preprocessed = frame_preprocessed / 255.0  # Normalize
    frame_preprocessed = np.expand_dims(frame_preprocessed, axis=-1)  # Add channel dimension
    frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)  # Add batch dimension
    prediction = synthetic_text_model.predict(frame_preprocessed)
    text = ""  # Implement conversion to text based on your model output
    print(f"Synthetic Text Recognition: {text}")  # Print recognized text
    return text

# Function to recognize handwritten text using a trained model
def recognize_handwritten_text(frame, handwritten_text_model):
    resized_frame = cv2.resize(frame, (128, 128))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame / 255.0
    input_data = np.expand_dims(normalized_frame, axis=0)
    input_data2 = np.zeros((1, 10))  # Adjust shape and content based on your model's requirement
    prediction = handwritten_text_model.predict([input_data, input_data2])
    text = ""  # Implement conversion to text based on your model output
    print(f"Handwritten Text Recognition: {text}")  # Print recognized text
    return text

# Function to recognize MNIST digits using a trained model
def recognize_mnist_digits(frame, mnist_model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_preprocessed = cv2.resize(gray_frame, (28, 28))  # Adjust size as per model requirements
    frame_preprocessed = np.expand_dims(frame_preprocessed, axis=-1)  # Add channel dimension
    frame_preprocessed = np.expand_dims(frame_preprocessed, axis=0)  # Add batch dimension
    prediction = mnist_model.predict(frame_preprocessed)
    text = ""  # Implement conversion to text based on your model output
    print(f"MNIST Digit Recognition: {text}")  # Print recognized text
    return text

# Main function
def main():
    # Load your trained models with adjusted paths
    try:
        synthetic_text_model = load_model(r"C:\Users\Gang\PycharmProjects\AMD\.venv\Trained Models\advanced_synthetic_text_cnn_model.h5")
        print("Synthetic text model loaded successfully.")
        handwritten_model = load_model(r"C:\Users\Gang\PycharmProjects\AMD\.venv\Trained Models\final_hand.h5")
        print("Handwritten text model loaded successfully.")
        mnist_model = load_model(r"C:\Users\Gang\PycharmProjects\AMD\.venv\Trained Models\advanced_mnist_cnn_model.h5")
        print("MNIST model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize the video capture
    cap = cv2.VideoCapture(0)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        ocr_text = ocr_from_frame(frame)

        if ocr_text:
            play_audio(ocr_text, lang='en')
        else:
            synthetic_text = recognize_synthetic_text(frame, synthetic_text_model)
            handwritten_text = recognize_handwritten_text(frame, handwritten_model)
            mnist_digits = recognize_mnist_digits(frame, mnist_model)

            if synthetic_text.strip():
                play_audio(synthetic_text, lang='en')
            elif handwritten_text.strip():
                play_audio(handwritten_text, lang='en')
            elif mnist_digits.strip():
                play_audio(mnist_digits, lang='en')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
