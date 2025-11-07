import sys
import os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

#!/usr/bin/env python3
"""
sentinel_smart_lock.py
Full integrated Sentinel Smart Lock:
- Fullscreen Tkinter GUI with camera preview (auto-detect)
- RFID (optional), Face recognition (optional)
- CSV logging, cooldown persistence, startup state recovery
- Add User button -> launches /home/project/Desktop/Att/add.py (auto-return after 2 minutes)
"""

import os
import sys
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

# GUI
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Try RPi GPIO + RFID; if not available, we'll keep running without hardware
try:
    import RPi.GPIO as GPIO
    from mfrc522 import SimpleMFRC522
except Exception:
    GPIO = None
    SimpleMFRC522 = None
    print("Warning: RPi.GPIO or mfrc522 not available (running off-Pi?)")

# ---------------- CONFIG ----------------
LOG_FILE = "access_log.csv"
COOLDOWN_FILE = "cooldown.json"
MASTER_TAG = "769839607204"
MASTER_NAME = "Universal"

RELAY_GPIO = 11
COOLDOWN_TIME = 10          # seconds per user
ADD_USER_TIMEOUT = 120      # seconds (2 minutes)
COOLDOWN_SAVE_INTERVAL = 5  # seconds

CAM_TRY_INDICES = list(range(0, 5))  # try these indices for camera
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ---------------- STATE + LOCKS ----------------
csv_mutex = Lock()
cooldown_mutex = Lock()
lock_mutex = Lock()
frame_mutex = Lock()

user_last_action = {}       # identifier -> last action timestamp
lock_open = False
current_user = None

# GUI-shared state dict
gui_state = {
    "rfid_text": "None",
    "face_text": "None",
    "lock_status": "Closed",
    "add_user_active": False,
    "last_user_activity": time.time(),
}

# Ensure CSV exists
columns = ["Method", "Identifier", "Open Timestamp", "Close Timestamp", "Detected Timestamp"]
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)

# ---------------- GPIO SETUP ----------------
def gpio_setup():
    if GPIO is None:
        return
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(RELAY_GPIO, GPIO.OUT)
    GPIO.output(RELAY_GPIO, GPIO.LOW)

gpio_setup()

# ---------------- LOGGING ----------------
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

# ---------------- COOLDOWN SAVE/LOAD ----------------
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

# ---------------- STARTUP RECOVERY ----------------
def check_last_lock_state():
    global lock_open, current_user
    try:
        df = pd.read_csv(LOG_FILE)
    except Exception:
        df = pd.DataFrame(columns=columns)
    if not df.empty:
        mask = (df["Open Timestamp"] != "") & (df["Close Timestamp"] == "")
        if mask.any():
            last = df[mask].iloc[-1]
            if GPIO is not None:
                try:
                    GPIO.output(RELAY_GPIO, GPIO.HIGH)
                except Exception:
                    pass
            lock_open = True
            current_user = last["Identifier"]
            with cooldown_mutex:
                user_last_action[current_user] = time.time()
            gui_state["lock_status"] = f"Opened by {current_user}"
            print(f"🔓 Lock opened on startup by {current_user} (last session not closed)")
            return
    # else closed
    if GPIO is not None:
        try:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
        except Exception:
            pass
    print("🔒 Lock closed on startup")

check_last_lock_state()

# ---------------- LOCK CONTROL ----------------
def open_lock(method, identifier):
    global lock_open, current_user
    with lock_mutex:
        last_time = user_last_action.get(identifier, 0)
        if time.time() - float(last_time) < COOLDOWN_TIME:
            # cooldown active
            return False
        if not lock_open:
            if GPIO is not None:
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
            gui_state["lock_status"] = f"Opened ({method})"
            gui_state["last_user_activity"] = time.time()
            print(f"🔓 Lock opened by {identifier}")
            return True
        return False

def close_lock():
    global lock_open, current_user
    with lock_mutex:
        if current_user is None:
            return False
        last_time = user_last_action.get(current_user, 0)
        if time.time() - float(last_time) < COOLDOWN_TIME:
            return False
        if lock_open:
            if GPIO is not None:
                try:
                    GPIO.output(RELAY_GPIO, GPIO.LOW)
                except Exception:
                    pass
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry("RFID" if "RFID" in str(current_user) else "FACE", current_user, close_time=now)
            print(f"🔒 Lock closed by {current_user}")
            with cooldown_mutex:
                user_last_action[current_user] = time.time()
            lock_open = False
            current_user = None
            gui_state["lock_status"] = "Closed"
            gui_state["last_user_activity"] = time.time()
            return True
        return False

# ---------------- RFID THREAD ----------------
reader = None
if SimpleMFRC522 is not None:
    try:
        reader = SimpleMFRC522()
    except Exception:
        reader = None
        print("RFID reader init failed")

def handle_rfid_detection(tag_id):
    tag_str = str(tag_id)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # MASTER tag behavior
    if tag_str == MASTER_TAG:
        last_time = user_last_action.get(MASTER_NAME, 0)
        if time.time() - float(last_time) < COOLDOWN_TIME:
            return
        if lock_open:
            # close
            close_lock()
            # log close with master
            try:
                df = pd.read_csv(LOG_FILE)
                mask = (df["Open Timestamp"] != "") & (df["Close Timestamp"] == "")
                if mask.any():
                    idx = df[mask].index[-1]
                    df.loc[idx, "Close Timestamp"] = now
                    df.loc[len(df)] = ["RFID", MASTER_NAME, "", "", now]
                    df.to_csv(LOG_FILE, index=False)
            except Exception:
                pass
        else:
            open_lock("RFID", MASTER_NAME)
            try:
                df = pd.read_csv(LOG_FILE)
                df.loc[len(df)] = ["RFID", MASTER_NAME, now, "", ""]
                df.to_csv(LOG_FILE, index=False)
            except Exception:
                pass
        with cooldown_mutex:
            user_last_action[MASTER_NAME] = time.time()
        gui_state["rfid_text"] = MASTER_NAME
        gui_state["last_user_activity"] = time.time()
        return

    # Normal RFID lookup
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
        print(f"⚠️ Unknown RFID {tag_str} detected — logged only")
        gui_state["rfid_text"] = f"Unknown({tag_str})"
        gui_state["last_user_activity"] = time.time()
        return

    name = res[0]
    identifier = f"{name}({tag_str})"
    gui_state["rfid_text"] = f"{tag_str} - {name}"
    gui_state["last_user_activity"] = time.time()

    if lock_open:
        if current_user == identifier:
            close_lock()
        else:
            print(f"📟 Detected {identifier} (lock already opened)")
            log_entry("RFID", identifier, detected_time=now)
    else:
        open_lock("RFID", identifier)

def rfid_thread_loop(stop_event: Event):
    if reader is None:
        print("RFID thread: reader not available, skipping.")
        return
    while not stop_event.is_set():
        try:
            uid, _ = reader.read_no_block()
            if uid:
                handle_rfid_detection(uid)
        except Exception:
            traceback.print_exc()
        time.sleep(0.2)

# ---------------- FACE RECOGNITION (uses shared frame) ----------------
detector = dlib.get_frontal_face_detector()
predictor = None
face_model = None
try:
    predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
    face_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
except Exception as e:
    predictor = None
    face_model = None
    print("Face model files missing or dlib init error:", e)

known_features = []
known_names = []
def load_face_db():
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            known_names.append(df.iloc[i, 0])
            feat = [float(df.iloc[i, j]) for j in range(1, 129)]
            known_features.append(feat)
load_face_db()

def face_thread_loop(stop_event: Event):
    global frame
    # We'll read frames from the shared `frame` produced by camera thread
    while not stop_event.is_set():
        if predictor is None or face_model is None or not known_features:
            time.sleep(0.5)
            continue
        with frame_mutex:
            local_frame = None if frame is None else frame.copy()
        if local_frame is None:
            time.sleep(0.05)
            continue
        try:
            # convert to BGR for dlib if necessary (our frame is RGB)
            bgr = cv2.cvtColor(local_frame, cv2.COLOR_RGB2BGR)
            faces = detector(bgr, 0)
            recognized = False
            for face in faces:
                shape = predictor(bgr, face)
                feature = np.array(face_model.compute_face_descriptor(bgr, shape))
                distances = [np.linalg.norm(feature - np.array(f)) for f in known_features]
                if distances and min(distances) < 0.6:
                    name = known_names[int(np.argmin(distances))]
                    gui_state["face_text"] = name
                    gui_state["last_user_activity"] = time.time()
                    # open/close logic
                    if lock_open and current_user == name:
                        last_time = user_last_action.get(name, 0)
                        if time.time() - float(last_time) >= COOLDOWN_TIME:
                            close_lock()
                    elif not lock_open:
                        open_lock("FACE", name)
                    else:
                        log_entry("FACE", name, detected_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    recognized = True
                    break
            if not recognized:
                gui_state["face_text"] = "Unknown"
        except Exception:
            traceback.print_exc()
        time.sleep(0.12)

# ---------------- PERIODIC COOLDOWN SAVE ----------------
def cooldown_saver_loop(stop_event: Event):
    while not stop_event.is_set():
        save_cooldown()
        time.sleep(COOLDOWN_SAVE_INTERVAL)

# ---------------- CAMERA THREAD (single capture) ----------------
camera_cap = None
def camera_init_and_stream(stop_event: Event):
    global camera_cap, frame
    # small warmup
    time.sleep(1.0)
    camera_cap = None
    for idx in CAM_TRY_INDICES:
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap.release()
                continue
            # set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
            camera_cap = cap
            print(f"Camera opened at index {idx}")
            break
        except Exception:
            continue
    if camera_cap is None:
        print("No camera available (tried indices {}). GUI will run without preview.".format(CAM_TRY_INDICES))
        return

    while not stop_event.is_set():
        ret, img = camera_cap.read()
        if ret:
            # convert to RGB for PIL
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with frame_mutex:
                frame = rgb
        else:
            time.sleep(0.05)

    try:
        camera_cap.release()
    except Exception:
        pass

# ---------------- GUI ----------------
class SentinelGUI:
    def __init__(self, root, stop_event: Event):
        self.root = root
        self.stop_event = stop_event
        root.title("Sentinel Smart Lock")
        root.attributes("-fullscreen", True)
        self.bg = "#071428"
        root.configure(bg=self.bg)

        # header
        self.header = tk.Label(root, text="🛡️ Sentinel Smart Lock", font=("Helvetica", 34, "bold"), bg=self.bg, fg="#7BE0E0")
        self.header.pack(pady=(12, 6))

        # datetime
        self.time_label = tk.Label(root, text="", font=("Helvetica", 16), bg=self.bg, fg="#CFEFF0")
        self.time_label.pack()

        # camera frame
        self.cam_frame = tk.Frame(root, bg="#07202A", bd=6, relief="ridge")
        self.cam_frame.pack(pady=16)
        self.camera_label = tk.Label(self.cam_frame)
        self.camera_label.pack()

        # status
        self.status_frame = tk.Frame(root, bg=self.bg)
        self.status_frame.pack(pady=(10, 10))
        self.rfid_label = tk.Label(self.status_frame, text="RFID Detected : None", font=("Helvetica", 18), bg=self.bg, fg="#A3FFD9")
        self.rfid_label.pack(pady=4)
        self.face_label = tk.Label(self.status_frame, text="Face Detected : None", font=("Helvetica", 18), bg=self.bg, fg="#A3FFD9")
        self.face_label.pack(pady=4)
        self.lock_label = tk.Label(self.status_frame, text="Lock Status   : Closed", font=("Helvetica", 18), bg=self.bg, fg="#FFD27A")
        self.lock_label.pack(pady=4)

        # add user button
        self.add_btn = tk.Button(root, text="➕ ADD USER", font=("Helvetica", 20, "bold"), bg="#1E7A6F", fg="white", padx=30, pady=10, command=self.on_add_user)
        self.add_btn.pack(pady=(14, 30))

        # footer
        self.footer = tk.Label(root, text="Developed by Vadeendra Karanam", font=("Helvetica", 14), bg=self.bg, fg="#9FB9BE")
        self.footer.pack(side="bottom", pady=8)

        # schedule updates
        self.update_clock()
        self.update_frame()
        self.update_status()
        self.check_add_user_timeout()

        # close binds
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Escape>", lambda e: self.on_close())

        self.add_proc = None  # subprocess Popen for add.py

    def update_clock(self):
        self.time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        if not self.stop_event.is_set():
            self.root.after(1000, self.update_clock)

    def update_frame(self):
        # read global frame and display resized appropriately
        with frame_mutex:
            local = None if frame is None else frame.copy()
        if local is not None:
            # compute desired size - take 55% width and 50% height of screen
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            w = int(sw * 0.55)
            h = int(sh * 0.5)
            try:
                img = Image.fromarray(local)
                img = img.resize((w, h))
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
        # hide main widgets and show a placeholder while launching add.py externally
        gui_state["add_user_active"] = True
        gui_state["last_user_activity"] = time.time()

        # hide everything
        for w in self.root.winfo_children():
            w.pack_forget()

        # show temporary message
        self.temp_label = tk.Label(self.root, text="🧍‍♂️ Add User Mode Active...\nClose the window or wait 2 minutes to return.",
                                   font=("Helvetica", 22, "bold"), bg=self.bg, fg="#66FFAA", justify="center")
        self.temp_label.pack(expand=True)

        # launch add.py non-blocking
        try:
            add_path = "/home/project/Desktop/Att/add.py"
            if os.path.exists(add_path):
                # Use Popen so we can launch without blocking
                self.add_proc = subprocess.Popen([sys.executable, add_path])
            else:
                print("add.py not found at", add_path)
        except Exception as e:
            print("Error launching add.py:", e)

        # start checking for return
        self.root.after(1000, self.check_add_user_close)

    def check_add_user_close(self):
        # if process ended -> return; or timeout -> return
        # Update last activity timestamp if add.py still running — we cannot know, so rely on timeout
        if self.add_proc is not None:
            ret = self.add_proc.poll()
            if ret is not None:
                # process finished
                gui_state["add_user_active"] = False

        # timeout check based on last_user_activity
        if time.time() - float(gui_state.get("last_user_activity", 0)) > ADD_USER_TIMEOUT:
            gui_state["add_user_active"] = False

        if not gui_state.get("add_user_active", False):
            # clear temp
            try:
                self.temp_label.pack_forget()
            except Exception:
                pass
            # rebuild full GUI by re-packing widgets (simplest approach: restart app window)
            # simpler: destroy and re-create root UI by exiting and re-running; here we'll restart python process
            print("Returning to main GUI (restarting UI)...")
            save_cooldown()
            os.execv(sys.executable, [sys.executable] + sys.argv)
            return

        # else check again
        if not self.stop_event.is_set():
            self.root.after(2000, self.check_add_user_close)

    def check_add_user_timeout(self):
        if gui_state.get("add_user_active"):
            if time.time() - float(gui_state.get("last_user_activity", 0)) > ADD_USER_TIMEOUT:
                gui_state["add_user_active"] = False
        if not self.stop_event.is_set():
            self.root.after(5000, self.check_add_user_timeout)

    def on_close(self):
        # cleanup and exit
        print("Exiting GUI...")
        gui_state["add_user_active"] = False
        save_cooldown()
        self.stop_event.set()
        try:
            if camera_cap is not None:
                camera_cap.release()
        except Exception:
            pass
        try:
            if GPIO is not None:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass
        os._exit(0)

# ---------------- MAIN START ----------------
def start_all():
    # event to signal threads to stop
    stop_event = Event()

    # Start camera thread (single capture used by GUI and face)
    cam_thread = Thread(target=camera_init_and_stream, args=(stop_event,), daemon=True)
    cam_thread.start()

    # Start face thread (reads shared frame)
    face_thread = Thread(target=face_thread_loop, args=(stop_event,), daemon=True)
    face_thread.start()

    # Start cooldown saver
    saver_thread = Thread(target=cooldown_saver_loop, args=(stop_event,), daemon=True)
    saver_thread.start()

    # Start RFID thread if reader present
    if reader is not None:
        r_thread = Thread(target=rfid_thread_loop, args=(stop_event,), daemon=True)
        r_thread.start()
    else:
        print("RFID reader not present - RFID thread skipped")

    # start GUI (blocking)
    root = tk.Tk()
    gui = SentinelGUI(root, stop_event)
    # attach stop_event to gui so on_close can set it
    gui.stop_event = stop_event
    root.mainloop()

    # on exit set stop_event and join threads
    stop_event.set()

if __name__ == "__main__":
    try:
        start_all()
    except KeyboardInterrupt:
        print("KeyboardInterrupt - exiting")
    except Exception:
        traceback.print_exc()
    finally:
        save_cooldown()
        if GPIO is not None:
            try:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
            except Exception:
                pass
