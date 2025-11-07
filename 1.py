import sys
import os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import os
import sys
import time
import datetime
import json
import sqlite3
import threading
from threading import Lock, Thread
import traceback

import pandas as pd
import numpy as np
import cv2
import dlib

# GUI
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# Raspberry Pi GPIO + RFID (may fail on non-RPi; handle gracefully)
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
COOLDOWN_TIME = 10  # seconds per user
ADD_USER_TIMEOUT = 120  # 2 minutes
COOLDOWN_SAVE_INTERVAL = 5  # seconds

# camera size (GUI will adapt)
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ---------------- STATE ----------------
csv_mutex = Lock()
cooldown_mutex = Lock()
lock_mutex = Lock()

user_last_action = {}         # identifier -> last action time
lock_open = False
current_user = None

# GUI-shared status (use thread-safe updates via locks if needed)
gui_state = {
    "rfid_text": "None",
    "face_text": "None",
    "lock_status": "Closed",
    "add_user_active": False,
    "last_user_activity": time.time(),
}

# ensure files exist
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

# ---------------- COOL DOWN PERSISTENCE ----------------
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

# ---------------- STARTUP STATE RECOVERY ----------------
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
                GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = last["Identifier"]
            with cooldown_mutex:
                user_last_action[current_user] = time.time()
            gui_state["lock_status"] = f"Opened by {current_user}"
            print(f"🔓 Lock opened on startup by {current_user} (last session not closed)")
        else:
            if GPIO is not None:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
            print("🔒 Lock closed on startup")
    else:
        print("🔒 Lock closed on startup")

check_last_lock_state()

# ---------------- LOCK CONTROL ----------------
def open_lock(method, identifier):
    global lock_open, current_user, user_last_action
    with lock_mutex:
        last_time = user_last_action.get(identifier, 0)
        if time.time() - float(last_time) < COOLDOWN_TIME:
            print(f"⏳ Cooldown active for {identifier} — ignoring open request")
            return
        if not lock_open:
            if GPIO is not None:
                GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = identifier
            with cooldown_mutex:
                user_last_action[identifier] = time.time()
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry(method, identifier, open_time=now)
            gui_state["lock_status"] = f"Opened ({method})"
            gui_state["last_user_activity"] = time.time()
            print(f"🔓 Lock opened by {identifier}")

def close_lock():
    global lock_open, current_user, user_last_action
    with lock_mutex:
        if current_user is None:
            return
        last_time = user_last_action.get(current_user, 0)
        if time.time() - float(last_time) < COOLDOWN_TIME:
            print(f"⏳ Cooldown active for {current_user} — ignoring close request")
            return
        if lock_open:
            if GPIO is not None:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry("RFID" if "RFID" in str(current_user) else "FACE", current_user, close_time=now)
            print(f"🔒 Lock closed by {current_user}")
            with cooldown_mutex:
                user_last_action[current_user] = time.time()
            lock_open = False
            current_user = None
            gui_state["lock_status"] = "Closed"
            gui_state["last_user_activity"] = time.time()

# ---------------- RFID THREAD ----------------
reader = None
if SimpleMFRC522 is not None:
    try:
        reader = SimpleMFRC522()
    except Exception:
        reader = None
        print("RFID reader init failed")

def handle_rfid_detection(tag_id):
    """Process a detected RFID tag (tag_id is integer or string)."""
    global lock_open, current_user
    tag_str = str(tag_id)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # MASTER TAG
    if tag_str == MASTER_TAG:
        last_time = user_last_action.get(MASTER_NAME, 0)
        if time.time() - float(last_time) < COOLDOWN_TIME:
            print(f"⏳ Cooldown active for {MASTER_NAME} — ignoring request")
            return
        if lock_open:
            print(f"📟 Detected {MASTER_NAME} (master) — closing lock...")
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
            if GPIO is not None:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
            print(f"🔒 Lock closed by {MASTER_NAME}")
            lock_open = False
            current_user = None
            gui_state["lock_status"] = "Closed"
        else:
            print(f"📟 Detected {MASTER_NAME} (master) — opening lock...")
            if GPIO is not None:
                GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = MASTER_NAME
            try:
                df = pd.read_csv(LOG_FILE)
                df.loc[len(df)] = ["RFID", MASTER_NAME, now, "", ""]
                df.to_csv(LOG_FILE, index=False)
            except Exception:
                pass
            print(f"🔓 Lock opened by {MASTER_NAME}")
            gui_state["lock_status"] = f"Opened ({MASTER_NAME})"
        with cooldown_mutex:
            user_last_action[MASTER_NAME] = time.time()
        gui_state["rfid_text"] = f"{MASTER_NAME}"
        gui_state["last_user_activity"] = time.time()
        return

    # Normal users: lookup in sqlite db
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

def rfid_thread_loop():
    if reader is None:
        print("RFID thread: reader not available, skipping.")
        return
    while True:
        try:
            uid, _ = reader.read_no_block()
            if uid:
                handle_rfid_detection(uid)
        except Exception:
            # reader may block or throw; continue gracefully
            traceback.print_exc()
        time.sleep(0.2)

# ---------------- FACE RECOGNITION THREAD ----------------
detector = dlib.get_frontal_face_detector()
predictor = None
face_model = None
try:
    predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
    face_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")
except Exception as e:
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

def face_thread_loop():
    # Keep face processing in background (used by GUI camera thread as well)
    # This thread will use a separate small capture to keep detection continuous even if GUI uses camera.
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            if predictor is None or face_model is None or not known_features:
                time.sleep(0.5)
                continue
            faces = detector(frame, 0)
            for face in faces:
                shape = predictor(frame, face)
                feature = np.array(face_model.compute_face_descriptor(frame, shape))
                distances = [np.linalg.norm(feature - np.array(f)) for f in known_features]
                if distances and min(distances) < 0.6:
                    name = known_names[int(np.argmin(distances))]
                    identifier = name
                    gui_state["face_text"] = name
                    gui_state["last_user_activity"] = time.time()
                    if lock_open and current_user == identifier:
                        last_time = user_last_action.get(identifier, 0)
                        if time.time() - float(last_time) >= COOLDOWN_TIME:
                            close_lock()
                    elif not lock_open:
                        open_lock("FACE", identifier)
                    else:
                        log_entry("FACE", identifier, detected_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        print(f"👤 Detected {identifier} (lock already opened)")
                else:
                    gui_state["face_text"] = "Unknown"
            time.sleep(0.1)
        except Exception:
            traceback.print_exc()
            time.sleep(0.5)

# ---------------- PERIODIC COOLDOWN SAVE ----------------
def cooldown_saver_loop():
    while True:
        save_cooldown()
        time.sleep(COOLDOWN_SAVE_INTERVAL)

# ---------------- GUI (Tkinter) ----------------
class SentinelGUI:
    def __init__(self, root):
        self.root = root
        root.title("Sentinel Smart Lock")
        root.attributes("-fullscreen", True)
        # Pleasant background color
        self.bg_color = "#071428"
        root.configure(bg=self.bg_color)

        # Header
        self.header = tk.Label(root, text="🛡️ Sentinel Smart Lock",
                               font=("Helvetica", 34, "bold"),
                               bg=self.bg_color, fg="#7BE0E0")
        self.header.pack(pady=(12, 4))

        # Date-Time
        self.time_label = tk.Label(root, text="", font=("Helvetica", 16),
                                   bg=self.bg_color, fg="#CFEFF0")
        self.time_label.pack()

        # Camera area frame with soft border
        self.cam_frame = tk.Frame(root, bg="#07202A", bd=6, relief="ridge")
        self.cam_frame.pack(pady=16)

        # Camera display
        self.camera_label = tk.Label(self.cam_frame)
        self.camera_label.pack()

        # Status fields
        self.status_frame = tk.Frame(root, bg=self.bg_color)
        self.status_frame.pack(pady=(10, 10))

        self.rfid_label = tk.Label(self.status_frame, text="RFID Detected : None",
                                   font=("Helvetica", 18), bg=self.bg_color, fg="#A3FFD9")
        self.rfid_label.pack(pady=4)

        self.face_label = tk.Label(self.status_frame, text="Face Detected : None",
                                   font=("Helvetica", 18), bg=self.bg_color, fg="#A3FFD9")
        self.face_label.pack(pady=4)

        self.lock_label = tk.Label(self.status_frame, text="Lock Status   : Closed",
                                   font=("Helvetica", 18), bg=self.bg_color, fg="#FFD27A")
        self.lock_label.pack(pady=4)

        # Add User Button
        self.add_btn = tk.Button(root, text="➕ ADD USER", font=("Helvetica", 20, "bold"),
                                 bg="#1E7A6F", fg="white", padx=30, pady=10,
                                 command=self.on_add_user)
        self.add_btn.pack(pady=(14, 30))

        # Footer
        self.footer = tk.Label(root, text="Developed by Vadeendra Karanam",
                               font=("Helvetica", 14), bg=self.bg_color, fg="#9FB9BE")
        self.footer.pack(side="bottom", pady=8)

        # Camera capture for GUI (single capture used for display)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

        # Schedule updates
        self.update_clock()
        self.update_frame()
        self.update_status()
        self.check_add_user_timeout()

        # Close protocol
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Escape>", lambda e: self.on_close())  # allow ESC to quit fullscreen

    def update_clock(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=now)
        self.root.after(1000, self.update_clock)

    def update_frame(self):
        try:
            ret, frame = self.cap.read()
            if ret:
                # Draw overlay text onto frame: title + datetime (optional)
                overlay = frame.copy()
                # header text on image
                cv2.putText(overlay, "Sentinel Smart Lock", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 255, 255), 2)
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(overlay, ts, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                # Convert BGR->RGB for Pillow
                cv2image = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
        except Exception:
            traceback.print_exc()
        finally:
            self.root.after(30, self.update_frame)

    def update_status(self):
        # Pull current gui_state and display
        self.rfid_label.config(text=f"RFID Detected : {gui_state.get('rfid_text', 'None')}")
        self.face_label.config(text=f"Face Detected : {gui_state.get('face_text', 'None')}")
        self.lock_label.config(text=f"Lock Status   : {gui_state.get('lock_status', 'Closed')}")
        self.root.after(500, self.update_status)

    def on_add_user(self):
        """Switch GUI to Add User mode — hide main screen, show add.py window."""
        gui_state["add_user_active"] = True
        gui_state["last_user_activity"] = time.time()

        # Hide main GUI
        for widget in self.root.winfo_children():
            widget.pack_forget()

        # Display temporary message
        self.temp_label = tk.Label(
            self.root,
            text="🧍‍♂️ Add User Mode Active...\nClose the window or wait 2 minutes to return.",
            font=("Helvetica", 22, "bold"),
            bg=self.bg_color,
            fg="#66FFAA"
        )
        self.temp_label.pack(expand=True)

        # Launch external add user GUI
        try:
            os.system("python3 /home/project/Desktop/Att/add.py &")
        except Exception as e:
            print("Error launching add.py:", e)

        # Start monitoring for return
        self.root.after(1000, self.check_add_user_close)

   
    def check_add_user_close(self):
        """Return to main GUI after add.py closes or after timeout."""
        # If timeout reached
        if time.time() - float(gui_state.get("last_user_activity", 0)) > ADD_USER_TIMEOUT:
            gui_state["add_user_active"] = False
            print("⏰ Add User timed out — returning to main screen")

        # If add user mode ended
        if not gui_state.get("add_user_active", False):
            # Clear temp message
            self.temp_label.pack_forget()

            # Rebuild the full GUI layout
            self.__init__(self.root)
            return

        # Re-check every 2 seconds
        self.root.after(2000, self.check_add_user_close)

    def check_add_user_timeout(self):
        if gui_state.get("add_user_active"):
            if time.time() - float(gui_state.get("last_user_activity", 0)) > ADD_USER_TIMEOUT:
                gui_state["add_user_active"] = False
                print("Add user timed out; returning to main screen")
        self.root.after(5000, self.check_add_user_timeout)


    def on_close(self):
        # cleanup
        try:
            gui_state["add_user_active"] = False
            save_cooldown()
            if self.cap and self.cap.isOpened():
                self.cap.release()
            if GPIO is not None:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
        except Exception:
            pass
        self.root.destroy()
        os._exit(0)  # ensure all threads stop

# ---------------- START THREADS & GUI ----------------
def start_background_threads():
    # RFID thread
    if reader is not None:
        t = Thread(target=rfid_thread_loop, daemon=True)
        t.start()
    else:
        print("RFID thread not started (no reader)")

    # face thread
    t2 = Thread(target=face_thread_loop, daemon=True)
    t2.start()

    # cooldown saver
    t3 = Thread(target=cooldown_saver_loop, daemon=True)
    t3.start()

if __name__ == "__main__":
    try:
        start_background_threads()
        root = tk.Tk()
        app = SentinelGUI(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("Exiting...")
    except Exception:
        traceback.print_exc()
    finally:
        save_cooldown()
        if GPIO is not None:
            try:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
            except Exception:
                pass
