import sys
import os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import dlib
import numpy as np
import cv2
import pandas as pd
import time
import sqlite3
import datetime
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

# --- Attendance database ---
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, time TEXT, date DATE, UNIQUE(name, date))")
conn.commit()
conn.close()

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

class FaceRFIDSystem:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.relay_triggered_time = 0
        self.relay_on = False
        self.rfid_tags = load_registered_rfid_tags()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        if not self.get_face_database():
            print("❌ Face database not found.")
            exit()

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                features = [csv_rd.iloc[i][j] for j in range(1, 129)]
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_features_known_list.append(features)
            return True
        return False

    def euclidean_distance(self, f1, f2):
        return np.linalg.norm(np.array(f1) - np.array(f2))

    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        if not cursor.fetchone():
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
            conn.commit()
            print(f"✅ {name} marked present at {current_time}")
        conn.close()

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            known_detected = False
            faces = detector(frame, 0)

            for face in faces:
                shape = predictor(frame, face)
                face_feature = face_reco_model.compute_face_descriptor(frame, shape)
                distances = [self.euclidean_distance(face_feature, f) for f in self.face_features_known_list]

                if distances and min(distances) < 0.6:
                    name = self.face_name_known_list[distances.index(min(distances))]
                    self.attendance(name)
                    known_detected = True
                else:
                    name = "unknown"

                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
                cv2.putText(frame, name, (face.left(), face.top() - 10), self.font, 0.6, (0, 255, 255), 1)

            # RFID check
            try:
                id, text = reader.read_no_block()
                if id and id in self.rfid_tags:
                    rfid_name = self.rfid_tags[id]
                    self.attendance(rfid_name)
                    known_detected = True
                    cv2.putText(frame, f"RFID: {rfid_name}", (10, 30), self.font, 0.7, (0, 255, 0), 2)
                    print(f"🔑 RFID authorized: {rfid_name}")
            except Exception:
                pass

            # Relay logic
            now = time.time()
            if known_detected and not self.relay_on:
                GPIO.output(RELAY_GPIO, RELAY_ON)
                self.relay_triggered_time = now
                self.relay_on = True
                print("🔓 Relay ON (Access Granted)")

            if self.relay_on and (now - self.relay_triggered_time >= 10):
                GPIO.output(RELAY_GPIO, RELAY_OFF)
                self.relay_on = False
                print("🔒 Relay OFF")

            # Show frame
            cv2.imshow("Face & RFID Recognizer", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()

if __name__ == "__main__":
    system = FaceRFIDSystem()
    system.run()
