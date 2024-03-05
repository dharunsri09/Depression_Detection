from flask import Flask, render_template, Response, request, redirect, url_for,send_from_directory
import cv2
import tensorflow as tf
import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend (non-interactive)
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import os
from transformers import pipeline
import requests
import sounddevice as sd
import soundfile as sf
from pydub import AudioSegment
from flask import Flask, render_template, request, send_file,url_for
import os
import uuid

import requests
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from transformers import pipeline


app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
emotion_classifier = pipeline("text-classification", model="transformersbook/distilbert-base-uncased-finetuned-emotion")
API_URL = "https://api-inference.huggingface.co/models/harshit345/xlsr-wav2vec-speech-emotion-recognition"
headers = {"Authorization": "Bearer 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"}


def query(filename):
    with open(filename, "rb") as f:
        data = f.read()
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def predict_emotion(text):
    prediction = emotion_classifier(text)[0]
    emotion=prediction['label']
    return emotion

emotions_text=[]


# Load the emotion detection model
model = tf.keras.models.load_model('best_model.h5')

# Define the emotions
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Depressed', 'Surprise', 'Neutral']
    
l=[]


def detect_emotion(save_on_stop=False):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Could not read frame from webcam.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                roi = roi_gray.astype('float') / 255.0
                roi = np.expand_dims(roi, axis=0)
                roi = np.expand_dims(roi, axis=-1)
                prediction = model.predict(roi)[0]
                label = emotions[prediction.argmax()]
                l.append(prediction.argmax())
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if save_on_stop:
                return redirect(url_for('show_distribution'))

    except Exception as e:
        print(f"Error in detect_emotion: {str(e)}")

    finally:
        end_time = time.time()
        elapsed_time_minutes = (end_time - start_time) / 60
        print(f"Total time taken: {elapsed_time_minutes:.2f} minutes")
        cap.release()
        cv2.destroyAllWindows()


@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Check login credentials (replace this with secure authentication logic)
        if username == 'dharuntkv@gmail.com' and password == 'Dharunsri99@':
            return redirect(url_for('index'))  # Redirect to homepage after successful login
        else:
            error = 'Invalid username or password. Please try again.'
    return render_template('login.html', error=error)

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_and_save')
def stop_and_save():
    detect_emotion(True)
    return "OK"

@app.route('/show_distribution')
def show_distribution():
    print(l)
    cv2.destroyAllWindows()
    emotion_counts = {emotion: l.count(index) for index, emotion in enumerate(emotions)}
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    plt.title('Emotion Distribution')
    plt.xlabel('Emotions')
    plt.ylabel('Count')
    image_stream = BytesIO()
    plt.savefig(image_stream, format='png')
    image_stream.seek(0)
    image_base64 = base64.b64encode(image_stream.read()).decode('utf-8')
    plt.close()
    return render_template('distribution.html', image_base64=image_base64)

@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    if text=="":
        with open('keylo.txt', 'r') as f:
            text = f.read(100)
            text1=list(map(str,text.split()))+f.read().split()
            for i in range(0, len(text1), 5):
                string = ' '.join(text1[i:i+5])
                l=[]
                l.append(string)
                emotion = predict_emotion(string)
                l.append(emotion)
                emotions_text.append(l)

            print("First 20 characters from the file:", text)
    file_path = "D:\\kongu\\Downloads\\DepressionDetection\\keylo.txt"
    if os.path.exists(file_path):
        os.remove(file_path)    
        print(f"File '{file_path}' deleted successfully.")
    else:
        print(f"File '{file_path}' does not exist.")
    emotion = predict_emotion(text)
    return render_template('result.html', text=text, emotion=emotion,emotions_text=emotions_text)


@app.route('/audio')
def audioDetect():
    print("Started")
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
    output = predict_emotion(text)
    return render_template('predicted.html', output=output)
def predict_emotion(text):
    prediction = emotion_classifier(text)[0]
    emotion=prediction['label']
    return emotion


@app.route('/record', methods=['POST'])
def record():
    if 'audio' not in request.files:
        return redirect(request.url)
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return redirect(request.url)
    
    if audio_file:
        audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'recorded_audio.wav'))
        return redirect(url_for('audioDetect'))

@app.route('/process')
def predictedAudio():
    audio_path = 'uploads/recorded_audio.wav'  # Update the file path
    
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)

    # Perform speech recognition
    try:
        text = recognizer.recognize_google(audio_data)
        print("Transcription:", text)
    except sr.UnknownValueError:
        text = "Speech recognition could not understand audio"
    except sr.RequestError as e:
        text = "Could not request results from Google Speech Recognition service"
    
    output = predict_emotion(text)
    return render_template('predicted.html', output=output)

def predict_emotion(text):
    prediction = emotion_classifier(text)[0]
    emotion = prediction['label']
    return emotion

def record_audio(duration, filename='recorded_audio.wav', sr=22050):
    print("Recording audio...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float64')
    sd.wait()  # Wait until recording is finished
    sf.write(filename, audio, sr)  # Save audio to file
    print("Audio recording saved as", filename)
if __name__ == '__main__':
    app.run(debug=True)
