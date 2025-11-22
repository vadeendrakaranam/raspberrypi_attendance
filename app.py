#!/usr/bin/env python3
# sentinel_smart_lock.py — Full integrated GUI + RFID + Face + Lock + Add User launcher

import sys
import os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import time
import datetime
import json
import sqlite3
import traceback
import subprocess
from threading import Thread, Lock, Event

import pandas as pd
import numpy as np
import cv2
import dlib
import tkinter as tk
from PIL import Image, ImageTk

frame = None
camera_cap = None  # global camera handle

try:
    import RPi.GPIO as GPIO
    from mfrc522 import SimpleMFRC522
except Exception:
    GPIO = None
    SimpleMFRC522 = None
    print("⚠️ Running without GPIO/RFID hardware support.")

# ---------------- CONFIG ----------------
LOG_FILE = "Logs.csv"
COOLDOWN_FILE = "cooldown.json"
MASTER_TAG = "769839607204"
MASTER_NAME = "Universal"
RELAY_GPIO = 11
TOGGLE_COOLDOWN = 12   # 12 seconds cooldown per user
GUI_DETECT_TIMEOUT = 1
CAM_TRY_INDICES = list(range(0, 5))
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ---------------- STATE ----------------
csv_mutex = Lock()
lock_mutex = Lock()
frame_mutex = Lock()

lock_open = False
current_user = None
user_last_toggle = {}   # tracks per-user last action time
active_rfid = {"text": "None", "last_seen": 0}
active_face = {"text": "None", "last_seen": 0}
unknown_last_seen = {}  # rate-limit unknown logs

# ---------------- GPIO Setup ----------------
def gpio_setup():
    if GPIO:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(RELAY_GPIO, GPIO.OUT)
        GPIO.output(RELAY_GPIO, GPIO.LOW)
gpio_setup()

# ---------------- Logging ----------------
def log_sentence(method, identifier, action):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p")
    sentence = f"{ts}: {action} by {method} user {identifier}."
    with csv_mutex:
        with open(LOG_FILE, "a") as f:
            f.write(sentence + "\n")
    print(sentence)

# ---------------- Cooldown ----------------
def save_cooldown():
    with Lock():
        try:
            with open(COOLDOWN_FILE, "w") as f:
                json.dump(user_last_toggle, f)
        except Exception as e:
            print("Error saving cooldown:", e)

def load_cooldown():
    global user_last_toggle
    if os.path.exists(COOLDOWN_FILE):
        try:
            with open(COOLDOWN_FILE, "r") as f:
                user_last_toggle = json.load(f)
        except Exception:
            user_last_toggle = {}
load_cooldown()

# ---------------- Lock Control ----------------
def open_lock(method, identifier):
    global lock_open, current_user
    now = time.time()
    last = user_last_toggle.get(identifier, 0)
    if now - last < TOGGLE_COOLDOWN:
        return False  # cooldown not finished

    with lock_mutex:
        if not lock_open:
            if GPIO:
                try: GPIO.output(RELAY_GPIO, GPIO.HIGH)
                except: pass
            lock_open = True
            current_user = identifier
            user_last_toggle[identifier] = now
            log_sentence(method, identifier, "Lock opened")
            return True
    return False

def close_lock(method, identifier):
    global lock_open, current_user
    now = time.time()
    last = user_last_toggle.get(identifier, 0)
    if now - last < TOGGLE_COOLDOWN:
        return False  # cooldown not finished

    with lock_mutex:
        if lock_open and current_user == identifier:
            if GPIO:
                try: GPIO.output(RELAY_GPIO, GPIO.LOW)
                except: pass
            lock_open = False
            user_last_toggle[identifier] = now
            log_sentence(method, identifier, "Lock closed")
            current_user = None
            return True
    return False

# ---------------- RFID ----------------
reader = None
if SimpleMFRC522:
    try:
        reader = SimpleMFRC522()
    except Exception:
        reader = None
        print("RFID init failed")

def handle_rfid_detection(tag_id):
    tag_str = str(tag_id)
    identifier = None
    now = time.time()

    # Master key
    if tag_str == MASTER_TAG:
        identifier = MASTER_NAME
        if not lock_open:
            open_lock("RFID", identifier)
        else:
            close_lock("RFID", identifier)
        active_rfid["text"] = identifier
        active_rfid["last_seen"] = now
        return

    # Check DB
    try:
        conn = sqlite3.connect("rfid_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?", (tag_str,))
        res = cursor.fetchone()
        conn.close()
    except Exception:
        res = None

    if not res:
        last_seen = unknown_last_seen.get(tag_str, 0)
        if now - last_seen > 5:
            unknown_last_seen[tag_str] = now
            active_rfid["text"] = f"Unknown({tag_str})"
            active_rfid["last_seen"] = now
            log_sentence("RFID", f"Unknown({tag_str})", "Detected")
        return

    # Known user
    name = res[0]
    identifier = f"{name}({tag_str})"
    active_rfid["text"] = identifier
    active_rfid["last_seen"] = now

    if not lock_open:
        open_lock("RFID", identifier)
    elif current_user == identifier:
        close_lock("RFID", identifier)
    else:
        log_sentence("RFID", identifier, "Detected")

def rfid_thread_loop(stop_event: Event):
    if reader is None:
        print("RFID reader not found.")
        return
    while not stop_event.is_set():
        try:
            uid, _ = reader.read_no_block()
            if uid:
                handle_rfid_detection(uid)
        except Exception:
            traceback.print_exc()
        time.sleep(0.25)

# ---------------- Face ----------------
detector = dlib.get_frontal_face_detector()
predictor, face_model = None, None
try:
    predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
    face_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
except Exception as e:
    print("Face model error:", e)

known_features, known_names = [], []
def load_face_db():
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            known_names.append(df.iloc[i, 0])
            known_features.append([float(df.iloc[i, j]) for j in range(1, 129)])
load_face_db()

def handle_face_detection(name):
    now = time.time()
    identifier = name if name != "Unknown" else "Unknown"
    active_face["text"] = identifier
    active_face["last_seen"] = now

    # Rate-limit unknown face logs
    if name == "Unknown":
        last_seen = unknown_last_seen.get("FACE_UNKNOWN", 0)
        if now - last_seen > 5:
            unknown_last_seen["FACE_UNKNOWN"] = now
            log_sentence("FACE", identifier, "Detected")
        return

    if not lock_open:
        open_lock("FACE", identifier)
    elif current_user == identifier:
        close_lock("FACE", identifier)
    else:
        log_sentence("FACE", identifier, "Detected")

def face_thread_loop(stop_event: Event):
    global frame
    while not stop_event.is_set():
        if predictor is None or face_model is None or not known_features:
            time.sleep(0.5)
            continue

        with frame_mutex:
            local = None if frame is None else frame.copy()
        if local is None:
            time.sleep(0.05)
            continue

        try:
            bgr = cv2.cvtColor(local, cv2.COLOR_RGB2BGR)
            faces = detector(bgr, 0)
            recognized = False
            for f in faces:
                shape = predictor(bgr, f)
                feat = np.array(face_model.compute_face_descriptor(bgr, shape))
                distances = [np.linalg.norm(feat - np.array(x)) for x in known_features]

                if distances and min(distances) < 0.6:
                    name = known_names[int(np.argmin(distances))]
                else:
                    name = "Unknown"

                handle_face_detection(name)
                recognized = True
                break

            if not recognized:
                active_face["text"] = "None"

        except Exception:
            traceback.print_exc()

        time.sleep(0.12)

# ---------------- Camera ----------------
def camera_init_and_stream(stop_event: Event):
    global camera_cap, frame
    for idx in CAM_TRY_INDICES:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                camera_cap = cap
                break
            else:
                cap.release()
        except Exception:
            continue
    if camera_cap is None:
        return

    while not stop_event.is_set():
        ret, img = camera_cap.read()
        if ret:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with frame_mutex:
                frame = rgb
        time.sleep(0.03)

    if camera_cap:
        camera_cap.release()
        camera_cap = None

# ---------------- GUI ----------------
class SentinelGUI:
    def __init__(self, root, stop_event: Event):
        self.root = root
        self.stop_event = stop_event
        self.fullscreen = True

        root.title("Sentinel Smart Lock")
        root.configure(bg="#071428")
        root.attributes("-fullscreen", True)
        root.focus_force()
        root.bind("<Escape>", self.toggle_fullscreen)

        self.header = tk.Label(root, text="🛡️ Sentinel Smart Lock",
                               font=("Helvetica", 34, "bold"), bg="#071428", fg="#7BE0E0")
        self.header.pack(pady=(12, 6))

        self.time_label = tk.Label(root, text="", font=("Helvetica", 16),
                                   bg="#071428", fg="#CFEFF0")
        self.time_label.pack()

        self.cam_frame = tk.Frame(root, bg="#07202A", bd=6, relief="ridge")
        self.cam_frame.pack(pady=(10, 10))
        self.camera_label = tk.Label(self.cam_frame)
        self.camera_label.pack()

        self.status_frame = tk.Frame(root, bg="#071428")
        self.status_frame.pack(pady=(10, 10))
        self.rfid_label = tk.Label(self.status_frame, text="RFID Detected : None",
                                   font=("Helvetica", 18), bg="#071428", fg="#A3FFD9")
        self.rfid_label.pack(pady=4)
        self.face_label = tk.Label(self.status_frame, text="Face Detected : None",
                                   font=("Helvetica", 18), bg="#071428", fg="#A3FFD9")
        self.face_label.pack(pady=4)
        self.lock_label = tk.Label(self.status_frame, text="Lock Status   : Closed",
                                   font=("Helvetica", 18), bg="#071428", fg="#FFD27A")
        self.lock_label.pack(pady=4)

        self.add_btn = tk.Button(root, text="➕ ADD USER", font=("Helvetica", 22, "bold"),
                                 bg="#13A88E", fg="white", padx=40, pady=14,
                                 command=self.on_add_user)
        self.add_btn.pack(pady=(20, 20))

        self.footer = tk.Label(root, text="Developed by Vadeendra Karanam [CSE-IoT 2026 Batch]",
                               font=("Helvetica", 14), bg="#071428", fg="#9FB9BE")
        self.footer.pack(side="bottom", pady=8)

        self.update_clock()
        self.update_frame()
        self.update_status()

        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)
        if not self.fullscreen:
            self.root.geometry("900x600+100+100")

    def update_clock(self):
        self.time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"))
        if not self.stop_event.is_set():
            self.root.after(1000, self.update_clock)

    def update_frame(self):
        with frame_mutex:
            local = None if frame is None else frame.copy()
        if local is not None:
            try:
                img = Image.fromarray(local).resize((320, 240))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            except Exception:
                traceback.print_exc()
        if not self.stop_event.is_set():
            self.root.after(30, self.update_frame)

    def update_status(self):
        now = time.time()
        if now - active_rfid["last_seen"] > GUI_DETECT_TIMEOUT:
            active_rfid["text"] = "None"
        if now - active_face["last_seen"] > GUI_DETECT_TIMEOUT:
            active_face["text"] = "None"

        self.rfid_label.config(text=f"RFID Detected : {active_rfid['text']}")
        self.face_label.config(text=f"Face Detected : {active_face['text']}")
        lock_text = f"Lock Status   : {'Opened by ' + current_user if lock_open else 'Closed'}"
        self.lock_label.config(text=lock_text)
        if not self.stop_event.is_set():
            self.root.after(500, self.update_status)

    def on_add_user(self):
        self.stop_event.set()
        if camera_cap: camera_cap.release()
        if GPIO: GPIO.output(RELAY_GPIO, GPIO.LOW)
        self.root.destroy()
        subprocess.Popen([sys.executable, "/home/project/Desktop/Att/add.py"])
        os._exit(0)

    def on_close(self):
        self.stop_event.set()
        save_cooldown()
        if camera_cap: camera_cap.release()
        if GPIO: GPIO.output(RELAY_GPIO, GPIO.LOW)
        self.root.destroy()
        os._exit(0)

# ---------------- START ----------------
def start_all():
    stop_event = Event()
    Thread(target=camera_init_and_stream, args=(stop_event,), daemon=True).start()
    Thread(target=face_thread_loop, args=(stop_event,), daemon=True).start()
    if reader:
        Thread(target=rfid_thread_loop, args=(stop_event,), daemon=True).start()
    root = tk.Tk()
    SentinelGUI(root, stop_event)
    root.mainloop()
    stop_event.set()

if __name__ == "__main__":
    start_all()
