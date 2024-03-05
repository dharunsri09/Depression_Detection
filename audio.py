
import requests
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from transformers import pipeline

emotion_classifier = pipeline("text-classification", model="transformersbook/distilbert-base-uncased-finetuned-emotion")
def record_audio(duration, filename='recorded_audio.wav', sr=22050):
    print("Recording audio...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float64')
    sd.wait()  # Wait until recording is finished
    sf.write(filename, audio, sr)  # Save audio to file
    print("Audio recording saved as", filename)

record_audio(duration=20)
audio_path = 'recorded_audio.wav'

# Initialize the recognizer
recognizer = sr.Recognizer()

# Load audio file
audio_file = "recorded_audio.wav"  # Replace "audio.wav" with the path to your audio file

# Load audio data
with sr.AudioFile(audio_file) as source:
    audio_data = recognizer.record(source)

# Perform speech recognition
try:
    text = recognizer.recognize_google(audio_data)
    print("Transcription:", text)
except sr.UnknownValueError:
    print("Speech recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
def predict_emotion(text):
    prediction = emotion_classifier(text)[0]
    emotion=prediction['label']
    return emotion
emotion = predict_emotion(text)
print(emotion)
