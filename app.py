import sys
import os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))
import time
import datetime
import pandas as pd
import sqlite3
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import cv2
import dlib
import numpy as np
from threading import Thread, Lock
import json
import subprocess
import tkinter as tk
from tkinter import Label, Button
from PIL import Image, ImageTk

# ---------------- GPIO SETUP ----------------
GPIO.setwarnings(False)
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)

# ---------------- FILES & CONFIG ----------------
LOG_FILE = "access_log.csv"
COOLDOWN_FILE = "cooldown.json"
columns = ["Method", "Identifier", "Open Timestamp", "Close Timestamp", "Detected Timestamp"]
csv_mutex = Lock()

MASTER_TAG = "769839607204"
MASTER_NAME = "Universal"

COOLDOWN_TIME = 10
user_last_action = {}
cooldown_mutex = Lock()

# ---------------- CSV INIT ----------------
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)

def log_entry(method, identifier, open_time="", close_time="", detected_time=""):
    with csv_mutex:
        df = pd.read_csv(LOG_FILE)
        if open_time:
            df.loc[len(df)] = [method, identifier, open_time, "", ""]
        elif close_time:
            mask = (df["Close Timestamp"] == "") & (df["Open Timestamp"] != "")
            if mask.any():
                idx = df[mask].index[-1]
                df.loc[idx, "Close Timestamp"] = close_time
            else:
                df.loc[len(df)] = [method, identifier, "", close_time, ""]
        elif detected_time:
            df.loc[len(df)] = [method, identifier, "", "", detected_time]
        df.to_csv(LOG_FILE, index=False)

# ---------------- GLOBAL LOCK STATE ----------------
lock_open = False
current_user = None
lock_mutex = Lock()

# ---------------- FACE & RFID SETUP ----------------
reader = SimpleMFRC522()
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

known_features, known_names = [], []
if os.path.exists("data/features_all.csv"):
    df = pd.read_csv("data/features_all.csv", header=None)
    for i in range(df.shape[0]):
        known_names.append(df.iloc[i, 0])
        known_features.append([df.iloc[i, j] for j in range(1, 129)])

# ---------------- GUI SETUP ----------------
root = tk.Tk()
root.title("Sentinel Smart Lock")
root.geometry("850x650")
root.configure(bg="#1a1a1a")

title_label = Label(root, text="🔐 Sentinel Smart Lock", font=("Arial", 22, "bold"), fg="cyan", bg="#1a1a1a")
title_label.pack(pady=10)

time_label = Label(root, text="", font=("Arial", 14), fg="white", bg="#1a1a1a")
time_label.pack()

camera_label = Label(root, bg="#1a1a1a")
camera_label.pack(pady=10)

rfid_label = Label(root, text="Detected RFID: None", font=("Arial", 13), fg="yellow", bg="#1a1a1a")
rfid_label.pack()

status_label = Label(root, text="Lock Status: Closed", font=("Arial", 13), fg="lightgreen", bg="#1a1a1a")
status_label.pack(pady=5)

opened_by_label = Label(root, text="Lock Opened by: None", font=("Arial", 13), fg="orange", bg="#1a1a1a")
opened_by_label.pack(pady=5)

def add_user():
    subprocess.Popen(["python3", "/home/project/Desktop/Att/add.py"])

Button(root, text="➕ Add User", font=("Arial", 12, "bold"), command=add_user, bg="green", fg="white", width=12).pack(pady=10)

# ---------------- TIME UPDATER ----------------
def update_time():
    time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    root.after(1000, update_time)

# ---------------- LOCK CONTROL ----------------
def open_lock(method, identifier):
    global lock_open, current_user
    GPIO.output(RELAY_GPIO, GPIO.HIGH)
    lock_open = True
    current_user = identifier
    status_label.config(text="Lock Status: 🔓 OPEN", fg="lime")
    opened_by_label.config(text=f"Lock Opened by: {method} {identifier}")
    log_entry(method, identifier, open_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def close_lock():
    global lock_open, current_user
    if current_user:
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        status_label.config(text="Lock Status: 🔒 CLOSED", fg="red")
        log_entry("RFID", current_user, close_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        current_user = None
        lock_open = False

# ---------------- RFID THREAD ----------------
def rfid_thread():
    while True:
        uid, _ = reader.read_no_block()
        if uid:
            tag_str = str(uid)
            rfid_label.config(text=f"Detected RFID: {tag_str}")
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            conn = sqlite3.connect("rfid_data.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?", (tag_str,))
            res = cursor.fetchone()
            conn.close()
            if not res:
                log_entry("RFID", f"Unknown({tag_str})", detected_time=now)
                opened_by_label.config(text=f"Lock Opened by: RFID Unknown")
                continue
            name = res[0]
            identifier = f"{name}({tag_str})"
            if not lock_open:
                open_lock("RFID", identifier)
            else:
                close_lock()
        time.sleep(0.2)

# ---------------- CAMERA THREAD ----------------
def camera_thread():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector(frame, 0)
        for face in faces:
            shape = predictor(frame, face)
            feature = face_model.compute_face_descriptor(frame, shape)
            if known_features:
                distances = [np.linalg.norm(np.array(feature) - np.array(f)) for f in known_features]
                if min(distances) < 0.6:
                    name = known_names[distances.index(min(distances))]
                    if not lock_open:
                        open_lock("FACE", name)
                    else:
                        log_entry("FACE", name, detected_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    opened_by_label.config(text="Lock Opened by: FACE Unknown")
                    log_entry("FACE", "UNKNOWN", detected_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
        time.sleep(0.05)
    cap.release()

# ---------------- START THREADS ----------------
Thread(target=rfid_thread, daemon=True).start()
Thread(target=camera_thread, daemon=True).start()
update_time()

root.mainloop()
