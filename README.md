Vision_A_Bot
Vision_A_Bot is a comprehensive project designed to aid visually impaired individuals by providing real-time object detection and text-to-speech conversion. This bot utilizes state-of-the-art AI models to recognize objects, read text aloud, and support multiple languages.

Features
Real-time Object Detection: Utilizes the YOLO model for accurate and efficient object detection.
Text Recognition: Supports English, Hindi, and Kannada text recognition.
Text-to-Speech (TTS): Converts recognized text to speech in multiple languages.
Language Detection: Automatically detects and reads text in the appropriate language.
Audio Feedback: Provides audio alerts for nearby objects.
AI Assistant: Integrated with the Gemini API for enhanced functionality.
Components
Hardware
3D Spec Model
Battery
Push Button
ESP32 Cam Module
Software
Python Scripts:
english.py: Text recognition and TTS for English.
hindi.py: Text recognition and TTS for Hindi.
kannada.py: Text recognition and TTS for Kannada.
yolo.py: Real-time object detection using YOLO model.
ai.py: AI assistant functionalities using Gemini API.
Installation
Clone the Repository:

git clone https://github.com/koushiksjain/Voice_A_Bot.git
cd Voice_A_Bot
Install Dependencies:
Ensure you have Python installed, then install the required packages:

pip install -r requirements.txt
Set Up Hardware:
Follow the instructions in the hardware_setup.md file to assemble the hardware components.

Usage
Run Object Detection:

python yolo.py
Run Text Recognition and TTS:
For English:

python english.py
For Hindi:

python hindi.py
For Kannada:

python kannada.py
Run AI Assistant:

python ai.py
Customization
Adding New Languages:
You can extend the text recognition and TTS capabilities by adding new language scripts similar to english.py, hindi.py, and kannada.py.

Integration with Other Models:
Modify the yolo.py script to integrate other object detection models as required.

Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

Acknowledgements
YOLO Model: YOLO
Gemini API: Gemini API
TTS Libraries: gTTS, pyttsx3
