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
COOLDOWN_TIME = 10  # seconds for toggle behavior
MASTER_TAG = "769839607204"
MASTER_NAME = "Universal"
RELAY_GPIO = 11
ADD_USER_TIMEOUT = 120
CAM_TRY_INDICES = list(range(0, 5))
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ---------------- STATE ----------------
csv_mutex = Lock()
lock_mutex = Lock()
frame_mutex = Lock()
lock_open = False
current_user = None

gui_state = {
    "face_text": "None",
    "lock_status": "Closed",
    "last_user_activity": time.time(),
}

active_rfid = {"text": ""}
active_face = {"text": ""}

user_last_toggle = {}  # Tracks last toggle per user for 10s toggle behavior

# ---------------- GPIO Setup ----------------
def gpio_setup():
    if GPIO:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(RELAY_GPIO, GPIO.OUT)
        GPIO.output(RELAY_GPIO, GPIO.LOW)
gpio_setup()

# ---------------- Lock Control ----------------
def open_lock(method, identifier):
    global lock_open, current_user
    with lock_mutex:
        if not lock_open:
            if GPIO:
                try:
                    GPIO.output(RELAY_GPIO, GPIO.HIGH)
                except Exception:
                    pass
            lock_open = True
            current_user = identifier
            gui_state["lock_status"] = f"Opened by {method}: {identifier}"
            gui_state["last_user_activity"] = time.time()
            print(f"🔓 Lock opened by {method}: {identifier}")
            return True
        return False

def close_lock():
    global lock_open, current_user
    with lock_mutex:
        if lock_open:
            if GPIO:
                try:
                    GPIO.output(RELAY_GPIO, GPIO.LOW)
                except Exception:
                    pass
            print(f"🔒 Lock closed by {current_user}")
            lock_open = False
            gui_state["lock_status"] = "Closed"
            gui_state["last_user_activity"] = time.time()
            current_user = None
            return True
        return False

# ---------------- Logging ----------------
def log_entry_sentence(method, identifier, action):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentence = f"{now}: {action} by {method} user {identifier}."
    with csv_mutex:
        with open(LOG_FILE, "a") as f:
            f.write(sentence + "\n")

# ---------------- RFID ----------------
reader = None
if SimpleMFRC522:
    try:
        reader = SimpleMFRC522()
    except Exception:
        reader = None
        print("RFID init failed")

def handle_rfid_detection(tag_id):
    global active_rfid, user_last_toggle
    tag_str = str(tag_id)
    now = time.time()

    # Determine identifier
    if tag_str == MASTER_TAG:
        identifier = MASTER_NAME
    else:
        try:
            conn = sqlite3.connect("rfid_data.db")
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?", (tag_str,))
            res = cursor.fetchone()
            conn.close()
        except Exception:
            res = None

        if not res:
            active_rfid["text"] = f"Unknown({tag_str})"
            log_entry_sentence("RFID", f"Unknown({tag_str})", "Detected")
            return
        identifier = f"{res[0]}({tag_str})"

    active_rfid["text"] = identifier
    gui_state["last_user_activity"] = now

    # Toggle after 10s since last toggle
    last = user_last_toggle.get(identifier, 0)
    if now - last < COOLDOWN_TIME:
        return

    if lock_open and current_user == identifier:
        close_lock()
        log_entry_sentence("RFID", identifier, "Lock closed")
    else:
        open_lock("RFID", identifier)
        log_entry_sentence("RFID", identifier, "Lock opened")

    user_last_toggle[identifier] = now

def rfid_thread_loop(stop_event: Event):
    global active_rfid
    if reader is None:
        print("RFID reader not found.")
        return
    while not stop_event.is_set():
        try:
            uid, _ = reader.read_no_block()
            if uid:
                handle_rfid_detection(uid)
            else:
                active_rfid["text"] = ""
        except Exception:
            traceback.print_exc()
        time.sleep(0.05)  # Fast loop for immediate detection

# ---------------- FACE ----------------
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

def face_thread_loop(stop_event: Event):
    global frame, current_user, lock_open, active_face, user_last_toggle
    while not stop_event.is_set():
        if predictor is None or face_model is None or not known_features:
            time.sleep(0.5)
            continue

        with frame_mutex:
            local = None if 'frame' not in globals() else (None if frame is None else frame.copy())
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
                    identifier = name
                    active_face["text"] = identifier
                    gui_state["last_user_activity"] = time.time()
                    recognized = True

                    now = time.time()
                    last = user_last_toggle.get(identifier, 0)
                    if now - last >= COOLDOWN_TIME:
                        if lock_open and current_user == identifier:
                            close_lock()
                            log_entry_sentence("FACE", identifier, "Lock closed")
                        else:
                            open_lock("FACE", identifier)
                            log_entry_sentence("FACE", identifier, "Lock opened")
                        user_last_toggle[identifier] = now
                    break

            if not recognized:
                active_face["text"] = ""

        except Exception:
            traceback.print_exc()
        time.sleep(0.05)

# ---------------- CAMERA ----------------
camera_cap = None
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

        self.footer = tk.Label(root, text="Developed by Vadeendra Karanam",
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
        self.time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
        rfid_display = active_rfid["text"] if active_rfid["text"] else "None"
        face_display = active_face["text"] if active_face["text"] else "None"
        self.rfid_label.config(text=f"RFID Detected : {rfid_display}")
        self.face_label.config(text=f"Face Detected : {face_display}")
        self.lock_label.config(text=f"Lock Status   : {gui_state.get('lock_status', 'Closed')}")
        if not self.stop_event.is_set():
            self.root.after(500, self.update_status)

    def on_add_user(self):
        print("🧍 Add User pressed — terminating main GUI and launching add.py...")
        self.stop_event.set()
        if camera_cap:
            camera_cap.release()
        if GPIO:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
        self.root.destroy()
        subprocess.Popen([sys.executable, "/home/project/Desktop/Att/add.py"])
        os._exit(0)

    def on_close(self):
        print("Exiting...")
        self.stop_event.set()
        if camera_cap:
            camera_cap.release()
        if GPIO:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
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
