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
import tkinter as tk
from PIL import Image, ImageTk

# ---------------- GPIO SETUP ----------------
GPIO.setwarnings(False)
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)  # initially closed

# ---------------- FILES & CONFIG ----------------
LOG_FILE = "access_log.csv"
COOLDOWN_FILE = "cooldown.json"
columns = ["Method", "Identifier", "Open Timestamp", "Close Timestamp", "Detected Timestamp"]
csv_mutex = Lock()

MASTER_TAG = "769839607204"
MASTER_NAME = "Universal"

# ---------------- COOL DOWN CONFIG ----------------
COOLDOWN_TIME = 10  # seconds per user
user_last_action = {}  # tracks last action time per user
cooldown_mutex = Lock()

# ---------------- CSV INIT ----------------
def init_csv():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)

init_csv()

# ---------------- LOGGING ----------------
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

# ---------------- LOCK STATE ----------------
lock_open = False
current_user = None
lock_mutex = Lock()
last_lock_method = ""

# ---------------- COOLDOWN ----------------
def save_cooldown():
    with cooldown_mutex:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(user_last_action, f)

def load_cooldown():
    global user_last_action
    if os.path.exists(COOLDOWN_FILE):
        with open(COOLDOWN_FILE, "r") as f:
            try:
                user_last_action = json.load(f)
            except:
                user_last_action = {}

# ---------------- LOCK FUNCTIONS ----------------
def open_lock(method, identifier):
    global lock_open, current_user, user_last_action, last_lock_method
    with lock_mutex:
        last_time = user_last_action.get(identifier, 0)
        if time.time() - last_time < COOLDOWN_TIME:
            return
        if not lock_open:
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = identifier
            last_lock_method = method
            with cooldown_mutex:
                user_last_action[identifier] = time.time()
            log_entry(method, identifier, open_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

def close_lock():
    global lock_open, current_user, user_last_action, last_lock_method
    with lock_mutex:
        if current_user is None:
            return
        last_time = user_last_action.get(current_user, 0)
        if time.time() - last_time < COOLDOWN_TIME:
            return
        if lock_open:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            log_entry(last_lock_method, current_user, close_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            with cooldown_mutex:
                user_last_action[current_user] = time.time()
            lock_open = False
            current_user = None
            last_lock_method = ""

# ---------------- STARTUP STATE ----------------
def check_last_lock_state():
    global lock_open, current_user
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        mask = (df["Open Timestamp"] != "") & (df["Close Timestamp"] == "")
        if mask.any():
            last = df[mask].iloc[-1]
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = last["Identifier"]
            with cooldown_mutex:
                user_last_action[current_user] = time.time()
        else:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
    else:
        GPIO.output(RELAY_GPIO, GPIO.LOW)

check_last_lock_state()
load_cooldown()

# ---------------- RFID SETUP ----------------
reader = SimpleMFRC522()
rfid_name_var = ""

def handle_rfid_detection(tag_id):
    global lock_open, current_user, user_last_action, rfid_name_var, last_lock_method

    tag_str = str(tag_id)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if tag_str == MASTER_TAG:
        last_time = user_last_action.get(MASTER_NAME, 0)
        if time.time() - last_time < COOLDOWN_TIME:
            return
        if lock_open:
            close_lock()
            rfid_name_var = f"{MASTER_NAME} (RFID) - Lock Closed"
        else:
            open_lock("RFID", MASTER_NAME)
            rfid_name_var = f"{MASTER_NAME} (RFID) - Lock Opened"
        with cooldown_mutex:
            user_last_action[MASTER_NAME] = time.time()
        return

    # Normal RFID users
    conn = sqlite3.connect("rfid_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?", (tag_str,))
    res = cursor.fetchone()
    conn.close()

    if not res:
        rfid_name_var = f"Unknown({tag_str})"
        return

    name = res[0]
    identifier = f"{name}({tag_str})"

    if lock_open:
        if current_user == identifier:
            close_lock()
            rfid_name_var = f"{identifier} (RFID) - Lock Closed"
        else:
            rfid_name_var = f"{identifier} (RFID) - Lock Already Open"
    else:
        open_lock("RFID", identifier)
        rfid_name_var = f"{identifier} (RFID) - Lock Opened"

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
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            known_names.append(df.iloc[i, 0])
            known_features.append([df.iloc[i, j] for j in range(1, 129)])

load_face_db()

frame_for_gui = None

def face_thread():
    global lock_open, current_user, user_last_action, rfid_name_var, frame_for_gui
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
                    identifier = name
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 0), 2)
                    cv2.putText(frame, name, (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                    if lock_open and current_user == identifier:
                        if time.time() - user_last_action.get(identifier,0) >= COOLDOWN_TIME:
                            close_lock()
                    elif not lock_open:
                        open_lock("FACE", identifier)
                    rfid_name_var = f"{identifier} (FACE) - {'Lock Opened' if lock_open else 'Lock Closed'}"
                else:
                    cv2.putText(frame, "Unknown", (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)
                    rfid_name_var = "Unknown FACE Detected"
        frame_for_gui = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        time.sleep(0.05)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Sentinel Smart Lock")
root.geometry("500x600")

time_label = tk.Label(root, text="", font=("Helvetica", 14))
time_label.pack(pady=5)

VIDEO_WIDTH, VIDEO_HEIGHT = 400, 300
video_label = tk.Label(root)
video_label.pack()

rfid_label = tk.Label(root, text="RFID/Face: Empty", font=("Helvetica", 12))
rfid_label.pack(pady=5)

lock_status_label = tk.Label(root, text="Lock is Closed", font=("Helvetica", 12, "bold"))
lock_status_label.pack(pady=5)

dev_label = tk.Label(root, text="Developed by Vadeendra Karanam", font=("Helvetica", 10))
dev_label.pack(side="bottom", pady=10)

def update_gui():
    time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    rfid_label.config(text=rfid_name_var if rfid_name_var else "RFID/Face: Empty")
    lock_status_label.config(text=f"Lock {'Opened' if lock_open else 'Closed'} by {current_user if current_user else 'None'} ({last_lock_method})")
    if frame_for_gui is not None:
        frame = cv2.resize(frame_for_gui, (VIDEO_WIDTH, VIDEO_HEIGHT))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    root.after(50, update_gui)

# ---------------- THREADS ----------------
t1 = Thread(target=rfid_thread, daemon=True)
t2 = Thread(target=face_thread, daemon=True)
t1.start()
t2.start()

update_gui()
root.mainloop()

# ---------------- CLEANUP ----------------
save_cooldown()
GPIO.output(RELAY_GPIO, GPIO.LOW)
