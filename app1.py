from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index3.html')

@app.route('/record', methods=['POST'])
def record():
    if 'audio' not in request.files:
        return redirect(request.url)
    
    audio_file = request.files['audio']
    if audio_file.filename == '':
        return redirect(request.url)
    
    if audio_file:
        audio_file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'recorded_audio.wav'))
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
