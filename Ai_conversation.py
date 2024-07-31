import speech_recognition as sr
from gtts import gTTS
import os
import pygame
import uuid
import sys
import google.generativeai as genai

# Set up Google Gemini API
GOOGLE_API_KEY = 'API_KEY'
genai.configure(api_key=GOOGLE_API_KEY)

generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    },
]

model = genai.GenerativeModel('gemini-1.5-flash',
                              generation_config=generation_config,
                              safety_settings=safety_settings)
convo = model.start_chat()

def text_to_speech(text, language='en'):
    filename = f"output_{uuid.uuid4()}.mp3"
    tts = gTTS(text=text, lang=language)
    tts.save(filename)
    os.chmod(filename, 0o666)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.quit()
    os.remove(filename)  # <--- Add this line to delete the file after playback

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone(device_index=0) as source:
        r.adjust_for_ambient_noise(source)
        print("Listening...")
        while True:
            try:
                audio = r.listen(source)
                text = r.recognize_google(audio).lower()
                print("You said: " + text)

                if text == "goodbye":
                    print("Stopping transcription and shutting down...")
                    sys.exit(0)

                convo.send_message(text)
                response = convo.last.text
                response = response.replace('#', '').replace('*', '')
                print("Vision: " + response)
                text_to_speech(response)

            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand your audio")
            except sr.RequestError as e:
                print("Could not request results from Google Speech Recognition service; {0}".format(e))

speech_to_text()
