import sys
import os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import cv2
import time
import sqlite3
import datetime
import threading
from flask import Flask, render_template, Response, redirect, request, url_for

import dlib
import numpy as np
import pandas as pd
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522

# --- Setup Dlib Models ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# --- Relay GPIO setup ---
RELAY_GPIO = 11
RELAY_ON = GPIO.HIGH
RELAY_OFF = GPIO.LOW

GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, RELAY_OFF)

# --- RFID setup ---
reader = SimpleMFRC522()

# --- Flask Setup ---
app = Flask(__name__)

# --- Load RFID users ---
def load_registered_rfid_tags():
    tags = {}
    conn = sqlite3.connect("rfid_tags.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS rfid_users (id INTEGER PRIMARY KEY AUTOINCREMENT, rfid_id INTEGER UNIQUE, name TEXT)")
    cursor.execute("SELECT rfid_id, name FROM rfid_users")
    rows = cursor.fetchall()
    for rfid_id, name in rows:
        tags[int(rfid_id)] = name
    conn.close()
    return tags

rfid_tags = load_registered_rfid_tags()

# --- Attendance Function ---
def mark_attendance(name):
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, time TEXT, date DATE, UNIQUE(name, date))")
    cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
    if not cursor.fetchone():
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
        conn.commit()
        print(f"✅ {name} marked present at {current_time}")
    conn.close()

# --- Video Capture ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

face_features_known_list = []
face_name_known_list = []
if os.path.exists("data/features_all.csv"):
    csv_rd = pd.read_csv("data/features_all.csv", header=None)
    for i in range(csv_rd.shape[0]):
        features = [csv_rd.iloc[i][j] for j in range(1, 129)]
        face_name_known_list.append(csv_rd.iloc[i][0])
        face_features_known_list.append(features)

last_rfid_id = None
last_rfid_time = 0
rfid_cooldown = 5
relay_triggered_time = 0
relay_on = False

def euclidean_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

# --- Video Generator ---
def gen_frames():
    global last_rfid_id, last_rfid_time, relay_on, relay_triggered_time

    while True:
        success, frame = cap.read()
        if not success:
            continue

        known_detected = False
        faces = detector(frame, 0)

        for face in faces:
            shape = predictor(frame, face)
            face_feature = face_reco_model.compute_face_descriptor(frame, shape)
            distances = [euclidean_distance(face_feature, f) for f in face_features_known_list]

            if distances and min(distances) < 0.6:
                name = face_name_known_list[distances.index(min(distances))]
                mark_attendance(name)
                known_detected = True
            else:
                name = "unknown"

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
            cv2.putText(frame, name, (face.left(), face.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        # RFID check
        try:
            id, text = reader.read_no_block()
            now = time.time()
            if id and id in rfid_tags:
                if id != last_rfid_id or (now - last_rfid_time) > rfid_cooldown:
                    rfid_name = rfid_tags[id]
                    mark_attendance(rfid_name)
                    known_detected = True
                    cv2.putText(frame, f"RFID: {rfid_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print(f"🔑 RFID authorized: {rfid_name}")
                    last_rfid_id = id
                    last_rfid_time = now
        except Exception:
            pass

        # Relay logic
        now = time.time()
        if known_detected and not relay_on:
            GPIO.output(RELAY_GPIO, RELAY_ON)
            relay_triggered_time = now
            relay_on = True
            print("🔓 Relay ON (Access Granted)")

        if relay_on and (now - relay_triggered_time >= 10):
            GPIO.output(RELAY_GPIO, RELAY_OFF)
            relay_on = False
            print("🔒 Relay OFF")

        # Encode Frame for Flask
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html', status="System Online")

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        print("👉 Place RFID card now...")
        id, text = reader.read()
        conn = sqlite3.connect("rfid_tags.db")
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO rfid_users (rfid_id, name) VALUES (?, ?)", (id, name))
        conn.commit()
        conn.close()
        print(f"✅ Registered RFID {id} with name {name}")
        return redirect(url_for('index'))
    return render_template('register.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
