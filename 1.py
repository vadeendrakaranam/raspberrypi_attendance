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

try:
    import RPi.GPIO as GPIO
    from mfrc522 import SimpleMFRC522
except Exception:
    GPIO = None
    SimpleMFRC522 = None
    print("⚠️ Running without GPIO/RFID hardware support.")

# ---------------- CONFIG ----------------
LOG_FILE = "access_log.csv"
COOLDOWN_FILE = "cooldown.json"
MASTER_TAG = "769839607204"
MASTER_NAME = "Universal"
RELAY_GPIO = 11
COOLDOWN_TIME = 10
ADD_USER_TIMEOUT = 120
COOLDOWN_SAVE_INTERVAL = 5
CAM_TRY_INDICES = list(range(0, 5))
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ---------------- STATE ----------------
csv_mutex = Lock()
cooldown_mutex = Lock()
lock_mutex = Lock()
frame_mutex = Lock()
user_last_action = {}
lock_open = False
current_user = None

gui_state = {
    "rfid_text": "None",
    "face_text": "None",
    "lock_status": "Closed",
    "add_user_active": False,
    "last_user_activity": time.time(),
}

columns = ["Method", "Identifier", "Open Timestamp", "Close Timestamp", "Detected Timestamp"]
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)

# ---------------- GPIO Setup ----------------
def gpio_setup():
    if GPIO:
        GPIO.setwarnings(False)
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(RELAY_GPIO, GPIO.OUT)
        GPIO.output(RELAY_GPIO, GPIO.LOW)
gpio_setup()

# ---------------- CSV Logging ----------------
def log_entry(method, identifier, open_time="", close_time="", detected_time=""):
    with csv_mutex:
        try:
            df = pd.read_csv(LOG_FILE)
        except Exception:
            df = pd.DataFrame(columns=columns)
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

# ---------------- Cooldown ----------------
def save_cooldown():
    with cooldown_mutex:
        try:
            with open(COOLDOWN_FILE, "w") as f:
                json.dump(user_last_action, f)
        except Exception as e:
            print("Error saving cooldown:", e)

def load_cooldown():
    global user_last_action
    if os.path.exists(COOLDOWN_FILE):
        try:
            with open(COOLDOWN_FILE, "r") as f:
                user_last_action = json.load(f)
        except Exception:
            user_last_action = {}
load_cooldown()

# ---------------- Lock Control ----------------
def open_lock(method, identifier):
    global lock_open, current_user
    with lock_mutex:
        last = user_last_action.get(identifier, 0)
        if time.time() - float(last) < COOLDOWN_TIME:
            return False
        if not lock_open:
            if GPIO:
                try:
                    GPIO.output(RELAY_GPIO, GPIO.HIGH)
                except Exception:
                    pass
            lock_open = True
            current_user = identifier
            with cooldown_mutex:
                user_last_action[identifier] = time.time()
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry(method, identifier, open_time=now)
            gui_state["lock_status"] = f"Opened by {method}: {identifier}"
            gui_state["last_user_activity"] = time.time()
            print(f"🔓 Lock opened by {method}: {identifier}")
            return True
        return False

def close_lock():
    global lock_open, current_user
    with lock_mutex:
        if not current_user:
            return False
        last = user_last_action.get(current_user, 0)
        if time.time() - float(last) < COOLDOWN_TIME:
            return False
        if lock_open:
            if GPIO:
                try:
                    GPIO.output(RELAY_GPIO, GPIO.LOW)
                except Exception:
                    pass
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry("AUTO", current_user, close_time=now)
            print(f"🔒 Lock closed by {current_user}")
            lock_open = False
            current_user = None
            gui_state["lock_status"] = "Closed"
            gui_state["last_user_activity"] = time.time()
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
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if tag_str == MASTER_TAG:
        if lock_open: close_lock()
        else: open_lock("RFID", MASTER_NAME)
        gui_state["rfid_text"] = MASTER_NAME
        gui_state["last_user_activity"] = time.time()
        return

    try:
        conn = sqlite3.connect("rfid_data.db")
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?", (tag_str,))
        res = cursor.fetchone()
        conn.close()
    except Exception:
        res = None

    if not res:
        log_entry("RFID", f"Unknown({tag_str})", detected_time=now)
        gui_state["rfid_text"] = f"Unknown({tag_str})"
        return

    name = res[0]
    identifier = f"{name}({tag_str})"
    gui_state["rfid_text"] = f"{tag_str} - {name}"
    gui_state["last_user_activity"] = time.time()
    if lock_open and current_user == identifier:
        close_lock()
    else:
        open_lock("RFID", name)

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

def face_thread_loop(stop_event: Event):
    global frame
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
                    gui_state["face_text"] = name
                    gui_state["last_user_activity"] = time.time()
                    if not lock_open:
                        open_lock("FACE", name)
                    recognized = True
                    break
            if not recognized:
                gui_state["face_text"] = "Unknown"
        except Exception:
            traceback.print_exc()
        time.sleep(0.12)

# ---------------- Camera ----------------
camera_cap, frame = None, None
def camera_init_and_stream(stop_event: Event):
    global camera_cap, frame
    for idx in CAM_TRY_INDICES:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
                camera_cap = cap
                print(f"Camera opened at index {idx}")
                break
        except Exception:
            continue
    if not camera_cap:
        print("No camera available.")
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

# ---------------- GUI ----------------
class SentinelGUI:
    def __init__(self, root, stop_event: Event):
        self.root, self.stop_event = root, stop_event
        root.title("Sentinel Smart Lock")
        root.attributes("-fullscreen", True)
        self.bg = "#071428"
        root.configure(bg=self.bg)

        self.header = tk.Label(root, text="🛡️ Sentinel Smart Lock", font=("Helvetica", 34, "bold"), bg=self.bg, fg="#7BE0E0")
        self.header.pack(pady=(12, 6))
        self.time_label = tk.Label(root, text="", font=("Helvetica", 16), bg=self.bg, fg="#CFEFF0")
        self.time_label.pack()

        self.cam_frame = tk.Frame(root, bg="#07202A", bd=6, relief="ridge")
        self.cam_frame.pack(pady=(10, 10))
        self.camera_label = tk.Label(self.cam_frame)
        self.camera_label.pack()

        self.status_frame = tk.Frame(root, bg=self.bg)
        self.status_frame.pack(pady=(10, 10))
        self.rfid_label = tk.Label(self.status_frame, text="RFID Detected : None", font=("Helvetica", 18), bg=self.bg, fg="#A3FFD9")
        self.rfid_label.pack(pady=4)
        self.face_label = tk.Label(self.status_frame, text="Face Detected : None", font=("Helvetica", 18), bg=self.bg, fg="#A3FFD9")
        self.face_label.pack(pady=4)
        self.lock_label = tk.Label(self.status_frame, text="Lock Status   : Closed", font=("Helvetica", 18), bg=self.bg, fg="#FFD27A")
        self.lock_label.pack(pady=4)

        self.add_btn = tk.Button(root, text="➕ ADD USER", font=("Helvetica", 22, "bold"), bg="#13A88E", fg="white",
                                 padx=40, pady=14, command=self.on_add_user)
        self.add_btn.pack(pady=(20, 20))

        self.footer = tk.Label(root, text="Developed by Vadeendra Karanam", font=("Helvetica", 14), bg=self.bg, fg="#9FB9BE")
        self.footer.pack(side="bottom", pady=8)

        self.update_clock()
        self.update_frame()
        self.update_status()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Escape>", lambda e: self.on_close())

    def update_clock(self):
        self.time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if not self.stop_event.is_set():
            self.root.after(1000, self.update_clock)

    def update_frame(self):
        with frame_mutex:
            local = None if frame is None else frame.copy()
        if local is not None:
            try:
                img = Image.fromarray(local)
                img = img.resize((320, 240))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            except Exception:
                traceback.print_exc()
        if not self.stop_event.is_set():
            self.root.after(30, self.update_frame)

    def update_status(self):
        self.rfid_label.config(text=f"RFID Detected : {gui_state.get('rfid_text', 'None')}")
        self.face_label.config(text=f"Face Detected : {gui_state.get('face_text', 'None')}")
        self.lock_label.config(text=f"Lock Status   : {gui_state.get('lock_status', 'Closed')}")
        if not self.stop_event.is_set():
            self.root.after(500, self.update_status)

    def on_add_user(self):
        print("🧍 Add User pressed — stopping system and launching add.py...")
        self.stop_event.set()
        try:
            if camera_cap:
                camera_cap.release()
            if GPIO:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
                GPIO.cleanup()
        except Exception:
            traceback.print_exc()
        try:
            self.root.destroy()
        except Exception:
            pass
        try:
            subprocess.call([sys.executable, "/home/project/Desktop/Att/add.py"])
        except Exception:
            traceback.print_exc()
        print("Returning to main GUI...")
        os.execv(sys.executable, [sys.executable] + sys.argv)

    def on_close(self):
        print("Exiting...")
        self.stop_event.set()
        save_cooldown()
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
    Thread(target=save_cooldown, daemon=True).start()
    if reader:
        Thread(target=rfid_thread_loop, args=(stop_event,), daemon=True).start()
    root = tk.Tk()
    SentinelGUI(root, stop_event)
    root.mainloop()
    stop_event.set()

if __name__ == "__main__":
    start_all()
