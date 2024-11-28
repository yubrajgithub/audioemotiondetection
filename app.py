import os
import sys
import warnings
from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from joblib import load
import mysql.connector
from werkzeug.security import generate_password_hash, check_password_hash

# Suppress TensorFlow and Scikit-learn warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# Set default encoding to UTF-8
sys.stdout.reconfigure(encoding='utf-8')

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with your secret key

# Database connection
db_config = {
    'user': 'root',
    'password': '',  # Default MySQL password is empty
    'host': '127.0.0.1',
    'database': 'flask_app'
}

# Load the model, scaler, and encoder
model = load_model('final_audio_emotion_model.h5')
scaler = load('scaler.joblib')
label_encoder = load('label_encoder.joblib')

# Define paths
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# File to store login state
LOGIN_STATE_FILE = 'login_state.txt'

# Utility functions to manage login state
def set_login_state(state):
    with open(LOGIN_STATE_FILE, 'w') as file:
        file.write('logged_in' if state else 'logged_out')

def is_logged_in():
    if not os.path.exists(LOGIN_STATE_FILE):
        return False
    with open(LOGIN_STATE_FILE, 'r') as file:
        return file.read().strip() == 'logged_in'

# Clear the login state at the start
set_login_state(False)

def extract_features(file_path):
    try:
        signal, sr = librosa.load(file_path, sr=44100)
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=signal).T, axis=0)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=signal, sr=sr).T, axis=0)
        mfcc = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
        rms = np.mean(librosa.feature.rms(y=signal).T, axis=0)
        mel = np.mean(librosa.feature.melspectrogram(y=signal, sr=sr).T, axis=0)
        return np.hstack([zcr, chroma_stft, mfcc, rms, mel])
    except Exception as e:
        print(f"Error in feature extraction: {e}")
        return None

def predict_emotion(file_path):
    features = extract_features(file_path)
    if features is None:
        return "Error in processing the audio file.", None
    features = scaler.transform([features])
    features = np.expand_dims(features, axis=2)
    y_pred = model.predict(features)
    y_pred_label = np.argmax(y_pred, axis=1)
    predicted_emotion = label_encoder.inverse_transform(y_pred_label)
    
    # Get confidence
    confidence = np.max(y_pred) * 100  # Percentage
    return predicted_emotion[0], confidence

def get_db_connection():
    conn = mysql.connector.connect(**db_config)
    return conn

@app.route('/')
def index():
    if is_logged_in():
        return render_template('index.html')
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file and file.filename.endswith('.wav'):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        emotion, confidence = predict_emotion(file_path)
        return render_template('result.html', emotion=emotion, confidence=confidence)
    return redirect(request.url)

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']
#         conn = get_db_connection()
#         cursor = conn.cursor(dictionary=True)
#         cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
#         user = cursor.fetchone()
#         cursor.close()
#         conn.close()
#         if user and check_password_hash(user['password'], password):
#             set_login_state(True)
#             return redirect(url_for('index'))
#         return 'Invalid credentials, please try again.'
#     return render_template('login.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE username = %s', (username,))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user and check_password_hash(user['password'], password):
            set_login_state(True)
            return redirect(url_for('index'))
        error = 'Invalid username or password. Please try again.'
    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        retype_password = request.form['retype_password']
        email = request.form['email']
        phone = request.form['phone']
        
        if password != retype_password:
            return 'Passwords do not match.'
        
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (username, password, email, phone) VALUES (%s, %s, %s, %s)',
                (username, hashed_password, email, phone)
            )
            conn.commit()
            return redirect(url_for('login'))
        except mysql.connector.IntegrityError:
            return 'Username or email already exists.'
        finally:
            cursor.close()
            conn.close()
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    set_login_state(False)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)
