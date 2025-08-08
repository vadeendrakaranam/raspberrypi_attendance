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
import tkinter as tk
from PIL import Image, ImageTk

# Dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Relay GPIO setup
RELAY_GPIO = 11
RELAY_ON = GPIO.HIGH   # Verify your relay logic (HIGH or LOW)
RELAY_OFF = GPIO.LOW

GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, RELAY_OFF)  # Relay off initially

# Initialize RFID reader
reader = SimpleMFRC522()

# Create SQLite attendance database and table
conn = sqlite3.connect("attendance.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS attendance (name TEXT, time TEXT, date DATE, UNIQUE(name, date))")
conn.commit()
conn.close()

# Load RFID registered tags from rfid_tags.db
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

class FaceRFIDApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face & RFID Recognizer")
        self.root.configure(bg="#222222")

        self.font = cv2.FONT_ITALIC
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.relay_triggered_time = 0
        self.relay_on = False
        self.rfid_tags = load_registered_rfid_tags()

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Load face database
        if not self.get_face_database():
            print("Failed to load face database. Exiting...")
            self.root.destroy()
            return

        # GUI: Time label
        self.time_label = tk.Label(root, font=("Arial", 16, "bold"), fg="#00ffcc", bg="#222222")
        self.time_label.pack(pady=5)

        # GUI: Video preview label
        self.video_label = tk.Label(root, bg="#444444", width=320, height=240)
        self.video_label.pack()

        self.update_time()
        self.process_frame()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                features = [csv_rd.iloc[i][j] for j in range(1, 129)]
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_features_known_list.append(features)
            return True
        else:
            print("'features_all.csv' not found!")
            return False

    def return_euclidean_distance(self, feature_1, feature_2):
        return np.linalg.norm(np.array(feature_1) - np.array(feature_2))

    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = sqlite3.connect("attendance.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
        if not cursor.fetchone():
            current_time = datetime.datetime.now().strftime('%H:%M:%S')
            cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
            conn.commit()
            print(f"{name} marked as present at {current_time}")
        conn.close()

    def update_time(self):
        now = time.strftime("%H:%M:%S")
        self.time_label.config(text=now)
        self.root.after(1000, self.update_time)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self.process_frame)
            return

        known_face_detected = False

        faces = detector(frame, 0)

        for face in faces:
            shape = predictor(frame, face)
            face_feature = face_reco_model.compute_face_descriptor(frame, shape)
            distances = [self.return_euclidean_distance(face_feature, f) for f in self.face_features_known_list]

            if distances and min(distances) < 0.6:
                name = self.face_name_known_list[distances.index(min(distances))]
                self.attendance(name)
                known_face_detected = True
            else:
                name = "unknown"

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
            cv2.putText(frame, name, (face.left(), face.top() - 10), self.font, 0.8, (0, 255, 255), 1)

        # RFID check (non-blocking)
        rfid_name = None
        try:
            id, text = reader.read_no_block()  # Implement read_no_block or use read with timeout if necessary
            if id in self.rfid_tags:
                rfid_name = self.rfid_tags[id]
                self.attendance(rfid_name)
                known_face_detected = True
                cv2.putText(frame, f"RFID: {rfid_name}", (10, 30), self.font, 0.7, (0, 255, 0), 2)
                print(f"RFID authorized: {rfid_name}")
        except Exception:
            pass

        current_time = time.time()

        # Relay control: ON for 10 seconds if authorized
        if known_face_detected:
            if not self.relay_on:
                GPIO.output(RELAY_GPIO, RELAY_ON)
                self.relay_triggered_time = current_time
                self.relay_on = True
                print("Relay ON (access granted)")

        if self.relay_on and (current_time - self.relay_triggered_time >= 10):
            GPIO.output(RELAY_GPIO, RELAY_OFF)
            self.relay_on = False
            print("Relay OFF")

        # Convert to Tkinter image and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(30, self.process_frame)

    def on_closing(self):
        self.cap.release()
        GPIO.cleanup()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRFIDApp(root)
    root.mainloop()
