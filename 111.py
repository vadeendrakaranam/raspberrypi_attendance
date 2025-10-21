import sys
import os
import time
import datetime
import sqlite3
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import cv2
import dlib
import numpy as np
from threading import Thread, Lock
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# ---------------- GPIO SETUP ----------------
GPIO.setwarnings(False)
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)

# ---------------- GLOBALS ----------------
MASTER_TAG = "769839607204"
MASTER_NAME = "Universal"
lock_open = False
current_user = None
last_detected_name = None
last_detected_method = None
lock_mutex = Lock()
rfid_mutex = Lock()

# ---------------- RFID SETUP ----------------
reader = SimpleMFRC522()

def handle_rfid_detection(tag_id):
    global lock_open, current_user, last_detected_name, last_detected_method
    tag_str = str(tag_id)
    identifier = None

    if tag_str == MASTER_TAG:
        identifier = MASTER_NAME
    else:
        conn = sqlite3.connect("rfid_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?", (tag_str,))
        res = cursor.fetchone()
        conn.close()
        if res:
            identifier = f"{res[0]}({tag_str})"
        else:
            identifier = f"Unknown({tag_str})"

    with lock_mutex:
        last_detected_name = identifier
        last_detected_method = "RFID"
        if identifier != f"Unknown({tag_str})":
            lock_open = True
            current_user = f"{identifier} (RFID)"
        else:
            lock_open = False
            current_user = None

def rfid_thread():
    while True:
        uid, _ = reader.read_no_block()
        if uid:
            handle_rfid_detection(uid)
        time.sleep(0.2)

# ---------------- FACE RECOGNITION ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
known_features, known_names = [], []

def load_face_db():
    import pandas as pd
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            known_names.append(df.iloc[i, 0])
            known_features.append([df.iloc[i, j] for j in range(1, 129)])
load_face_db()

def handle_face_detection(frame):
    global lock_open, current_user, last_detected_name, last_detected_method
    faces = detector(frame, 0)
    for face in faces:
        shape = predictor(frame, face)
        feature = face_model.compute_face_descriptor(frame, shape)
        if known_features:
            distances = [np.linalg.norm(np.array(feature) - np.array(f)) for f in known_features]
            if min(distances) < 0.6:
                name = known_names[distances.index(min(distances))]
                last_detected_name = name
                last_detected_method = "Face"
                lock_open = True
                current_user = f"{name} (Face)"
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 0), 2)
                cv2.putText(frame, name, (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
            else:
                last_detected_name = "Unknown"
                last_detected_method = "Face"
                lock_open = False
                current_user = None
    return frame

# ---------------- GUI ----------------
class AccessGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel Smart Lock")
        self.root.geometry("700x600")

        self.time_label = tk.Label(root, text="", font=("Helvetica", 16))
        self.time_label.pack(pady=5)

        self.camera_label = tk.Label(root)
        self.camera_label.pack()

        self.name_label = tk.Label(root, text="No detection yet", font=("Helvetica", 14))
        self.name_label.pack(pady=5)

        self.lock_status_label = tk.Label(root, text="Lock Closed", font=("Helvetica", 14))
        self.lock_status_label.pack(pady=5)

        self.add_user_btn = ttk.Button(root, text="Add User", command=self.add_user)
        self.add_user_btn.pack(pady=5)

        self.footer_label = tk.Label(root, text="Developed by Vadeendra Karanam", font=("Helvetica", 10))
        self.footer_label.pack(side=tk.BOTTOM, pady=5)

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        self.update_gui()

    def add_user(self):
        print("Add User clicked!")  # Implement Add User logic

    def update_gui(self):
        # Update time
        self.time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Camera preview
        ret, frame = self.cap.read()
        if ret:
            frame = handle_face_detection(frame)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

        # Show last detected name
        if last_detected_name:
            self.name_label.config(text=f"{last_detected_name} ({last_detected_method})")
        else:
            self.name_label.config(text="No detection yet")

        # Lock status
        if lock_open and current_user:
            self.lock_status_label.config(text=f"Lock Opened by: {current_user}")
        else:
            self.lock_status_label.config(text="Lock Closed")

        self.root.after(50, self.update_gui)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    t1 = Thread(target=rfid_thread, daemon=True)
    t1.start()
    root = tk.Tk()
    gui = AccessGUI(root)
    root.mainloop()
    GPIO.output(RELAY_GPIO, GPIO.LOW)
