import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import speech_recognition as sr
from flask import Flask, render_template, Response
import threading

app = Flask(__name__)

# Load emotion recognition model
emotion_model = tf.keras.models.load_model("emotion_model.h5")  # Replace with your model path
emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']

# Initialize MediaPipe Pose for hand movements
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize Speech Recognition
recognizer = sr.Recognizer()

# Initialize global variables
current_action = ""

def recognize_speech():
    """Function to capture speech and identify specific words like 'hi', 'super'"""
    global current_action
    with sr.Microphone() as source:
        print("Listening for speech...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio).lower()
            if 'hi' in text:
                current_action = "Saying 'Hi'"
            elif 'super' in text:
                current_action = "Saying 'Super'"
            print(f"Detected Speech: {current_action}")
        except Exception as e:
            print("Speech recognition failed:", e)

def recognize_emotion(frame):
    """Function to recognize facial emotion"""
    global current_action
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Get the first detected face
        face = frame[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))  # Resize to fit emotion model input
        face_resized = face_resized / 255.0
        face_resized = np.expand_dims(face_resized, axis=0)
        face_resized = np.expand_dims(face_resized, axis=-1)

        emotion_preds = emotion_model.predict(face_resized)
        emotion_idx = np.argmax(emotion_preds)
        current_action = f"Emotion Detected: {emotion_labels[emotion_idx]}"
        print(current_action)

def detect_pose(frame):
    """Function to detect pose and gestures (e.g., hand movements)"""
    global current_action
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)
    
    if results.pose_landmarks:
        # Example: Check if hands are raised
        left_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
        right_hand = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
        
        if left_hand.y < 0.3 and right_hand.y < 0.3:  # Threshold based on webcam resolution
            current_action = "Hands Raised"
            print(current_action)

def gen_frames():
    """Generate frames for the webcam stream and process each frame"""
    global current_action
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect Pose (hand movement, body gestures)
        detect_pose(frame)

        # Detect Emotion (face)
        recognize_emotion(frame)
        
        # Add text feedback for recognized actions
        cv2.putText(frame, f"Action: {current_action}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Render the webpage"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_recognition', methods=['POST'])
def start_recognition():
    """Start speech recognition in a separate thread"""
    threading.Thread(target=recognize_speech, daemon=True).start()
    return "Started listening for speech", 200

@app.route('/stop_feed', methods=['POST'])
def stop_feed():
    """Stop the webcam feed"""
    global cap
    if cap.isOpened():
        cap.release()
    return "Feed Stopped", 200

if __name__ == "__main__":
    app.run(debug=True)
