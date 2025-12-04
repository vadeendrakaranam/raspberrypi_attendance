#!/usr/bin/env python3
# sentinel_smart_lock_motor.py
# Full integrated GUI + RFID + Face detection + Motor (GPIO17) control (Option A: RFID OR Face enables motor GUI)
# Updated: continuous Machine ON/OFF status display and motor GUI green when detected, black when not
# Added: 6-second popup alerts for allowed / not allowed access

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
camera_cap = None

# optional hardware libs
try:
    import RPi.GPIO as GPIO
    from mfrc522 import SimpleMFRC522
except Exception:
    GPIO = None
    SimpleMFRC522 = None
    print("⚠ Running without GPIO/RFID hardware support.")

# ---------------- CONFIG ----------------
LOG_FILE = "Machine_Logs.csv"
COOLDOWN_FILE = "cooldown.json"   # kept for compatibility (not used for motor)
MASTER_TAG = "769839607204"
MASTER_NAME = "Admin"
MOTOR_GPIO = 17            # BCM pin (physical 11)
GUI_DETECT_TIMEOUT = 1.0
CAM_TRY_INDICES = list(range(0, 5))
CAM_WIDTH = 640
CAM_HEIGHT = 480

# ---------------- STATE & LOCKS ----------------
csv_mutex = Lock()
frame_mutex = Lock()
det_mutex = Lock()  # protects current_user / auth_method
machine_mutex = Lock()

current_user = None     # identifier string, set when known RFID or face detected
auth_method = None      # "RFID" or "FACE"
active_rfid = {"text": "None", "last_seen": 0}
active_face = {"text": "None", "last_seen": 0}
unknown_last_seen = {}

# Machine state tracking
machine_state = "OFF"           # "ON" or "OFF"
last_machine_user = None        # string identifier for last access
last_machine_method = None      # "RFID" or "FACE"

# ---------------- Email / Unknown Image Setup ----------------
import smtplib
from email.message import EmailMessage
import ssl

SENDER_EMAIL = "karanam.vadeendra123456@gmail.com"
APP_PASSWORD = "ibae dfoh tdlo uoow"
RECEIVER_EMAIL = "126158026@sastra.ac.in"

UNKNOWN_DIR = "unknown"
os.makedirs(UNKNOWN_DIR, exist_ok=True)

def send_email_alert(subject, body, image_path):
    try:
        msg = EmailMessage()
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL
        msg["Subject"] = subject
        msg.set_content(body)
        try:
            with open(image_path, "rb") as f:
                img = f.read()
            msg.add_attachment(img, maintype="image", subtype="jpeg", filename=os.path.basename(image_path))
        except Exception as e:
            print("Warning: failed to attach image:", e)
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, APP_PASSWORD)
            server.send_message(msg)
        try:
            os.remove(image_path)
        except Exception:
            pass
        print(f"Email sent and image removed: {image_path}")
    except Exception as e:
        print("Email sending failed:", e)

def capture_and_email(detect_type):
    ts = datetime.datetime.now().strftime("%Y-%m-%d_%I-%M-%S_%p")
    filename = f"{UNKNOWN_DIR}/unknown_{detect_type}_{ts}.jpg"
    with frame_mutex:
        local = None if frame is None else frame.copy()
    if local is None:
        print("No frame to capture")
        return
    try:
        cv2.imwrite(filename, cv2.cvtColor(local, cv2.COLOR_RGB2BGR))
    except Exception as e:
        print("Failed saving image:", e)
        return
    subject = f"Unknown {detect_type} Detected - {datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}"
    body = f"An unknown {detect_type.lower()} was detected on Sentinel Smart Lock.\nTime: {datetime.datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')}"
    send_email_alert(subject, body, filename)

# ---------------- GPIO Setup (motor only) ----------------
def gpio_setup():
    if GPIO:
        try:
            GPIO.setwarnings(False)
            GPIO.setmode(GPIO.BCM)   # use BCM numbering for MOTOR_GPIO
            GPIO.setup(MOTOR_GPIO, GPIO.OUT)
            GPIO.output(MOTOR_GPIO, GPIO.LOW)
            print(f"[GPIO] MOTOR GPIO {MOTOR_GPIO} initialized LOW")
        except Exception as e:
            print("GPIO setup error:", e)
    else:
        print("GPIO not available — running in simulation mode (no hardware toggles).")
gpio_setup()

# ---------------- Logging ----------------
def log_machine_access(action, identifier, method):
    line = f"Machine {action} accessed by {identifier} - {method}"
    with csv_mutex:
        try:
            with open(LOG_FILE, "a") as f:
                f.write(f"{datetime.datetime.now()}, {line}\n")
        except Exception as e:
            print("Failed writing log:", e)
    print(line)

# ---------------- RFID ----------------
reader = None
if SimpleMFRC522:
    try:
        reader = SimpleMFRC522()
    except Exception:
        reader = None
        print("RFID init failed")

def handle_rfid_detection(tag_id):
    global current_user, auth_method
    tag_str = str(tag_id)
    now = time.time()

    if tag_str == MASTER_TAG:
        with det_mutex:
            current_user = MASTER_NAME
            auth_method = "RFID"
            active_rfid["text"] = MASTER_NAME
            active_rfid["last_seen"] = now
        # popup for master
        try:
            gui_ref.show_popup("You are allowed to access the machine", "#13A88E")
        except Exception:
            pass
        return

    try:
        conn = sqlite3.connect("rfid_data.db")
        cur = conn.cursor()
        cur.execute("SELECT name FROM rfid_users WHERE tag_id=?", (tag_str,))
        res = cur.fetchone()
        conn.close()
    except Exception:
        res = None

    if not res:
        last_seen = unknown_last_seen.get(tag_str, 0)
        if now - last_seen > 5:
            unknown_last_seen[tag_str] = now
            active_rfid["text"] = f"Unknown({tag_str})"
            active_rfid["last_seen"] = now
            log_machine_access("Detected UNKNOWN RFID", f"Unknown({tag_str})", "RFID")
            Thread(target=capture_and_email, args=("RFID",), daemon=True).start()
            # popup for unknown RFID
            try:
                gui_ref.show_popup("You are NOT allowed to access the machine", "#C92424")
            except Exception:
                pass
        return

    name = res[0]
    with det_mutex:
        current_user = name
        auth_method = "RFID"
    active_rfid["text"] = name
    active_rfid["last_seen"] = now
    print(f"[RFID] Known user detected: {name}")
    # popup for known RFID
    try:
        gui_ref.show_popup("You are allowed to access the machine", "#13A88E")
    except Exception:
        pass

def rfid_thread_loop(stop_event: Event):
    if reader is None:
        print("RFID reader not found; skipping RFID thread.")
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
    predictor = None
    face_model = None
    print("Face models not loaded:", e)

known_features, known_names = [], []
def load_face_db():
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            known_names.append(df.iloc[i, 0])
            known_features.append([float(df.iloc[i, j]) for j in range(1, 129)])
load_face_db()

def handle_face_detection(name):
    global current_user, auth_method
    now = time.time()
    active_face["text"] = name
    active_face["last_seen"] = now

    if name == "Unknown":
        last_seen = unknown_last_seen.get("FACE_UNKNOWN", 0)
        if now - last_seen > 5:
            unknown_last_seen["FACE_UNKNOWN"] = now
            log_machine_access("Unknown Face Detected", "Unknown", "FACE")
            Thread(target=capture_and_email, args=("FACE",), daemon=True).start()
            # popup for unknown face
            try:
                gui_ref.show_popup("You are NOT allowed to access the machine", "#C92424")
            except Exception:
                pass
        return

    with det_mutex:
        current_user = name
        auth_method = "FACE"
    print(f"[FACE] Known user detected: {name}")
    # popup for known face
    try:
        gui_ref.show_popup("You are allowed to access the machine", "#13A88E")
    except Exception:
        pass

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
            for f in faces:
                shape = predictor(bgr, f)
                feat = np.array(face_model.compute_face_descriptor(bgr, shape))
                distances = [np.linalg.norm(feat - np.array(x)) for x in known_features]
                if distances and min(distances) < 0.6:
                    name = known_names[int(np.argmin(distances))]
                else:
                    name = "Unknown"
                handle_face_detection(name)
                break
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
        print("Camera not found / couldn't open.")
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

        root.title("HYBRID AUTHENTICATION UNIT")
        root.configure(bg="#071428")
        root.attributes("-fullscreen", True)
        root.focus_force()
        root.bind("<Escape>", self.toggle_fullscreen)

        # Header + time
        self.header = tk.Label(root, text="🛡 HYBRID AUTHENTICATION UNIT",
                               font=("Helvetica", 32, "bold"), bg="#071428", fg="#7BE0E0")
        self.header.pack(pady=(12,6))

        self.time_label = tk.Label(root, text="", font=("Helvetica", 16), bg="#071428", fg="#CFEFF0")
        self.time_label.pack()

        main = tk.Frame(root, bg="#071428")
        main.pack(expand=True, fill="both", padx=12, pady=10)

        # Camera panel
        cam_panel = tk.Frame(main, bg="#07202A", bd=6, relief="ridge")
        cam_panel.pack(side="left", padx=(6,12))
        self.camera_label = tk.Label(cam_panel)
        self.camera_label.pack()

        # Right panel - status + motor + add
        right = tk.Frame(main, bg="#071428")
        right.pack(side="left", fill="y", padx=(12,24))

        self.rfid_label = tk.Label(right, text="RFID Detected : None", font=("Helvetica", 18), bg="#071428", fg="#A3FFD9")
        self.rfid_label.pack(anchor="w", pady=4)
        self.face_label = tk.Label(right, text="Face Detected : None", font=("Helvetica", 18), bg="#071428", fg="#A3FFD9")
        self.face_label.pack(anchor="w", pady=4)

        self.status_label = tk.Label(right, text="Status : Waiting", font=("Helvetica", 16), bg="#071428", fg="#FFD27A")
        self.status_label.pack(anchor="w", pady=(10,8))

        # Motor GUI
        self.motor_frame = tk.Frame(right, bg="black", bd=4, relief="ridge")
        self.motor_frame.pack(pady=8, fill="x")
        self.motor_title = tk.Label(self.motor_frame, text="Machine Control", font=("Helvetica", 16, "bold"), bg="black", fg="#FFCF7A")
        self.motor_title.pack(pady=(8,6))

        self.motor_row = tk.Frame(self.motor_frame, bg="black")
        self.motor_row.pack(pady=(6,12))
        self.btn_on = tk.Button(self.motor_row, text="MACHINE ON", font=("Helvetica", 14, "bold"), bg="#1CBF4A", fg="black", padx=18, pady=8, state="disabled", command=self.machine_on)
        self.btn_on.pack(side="left", padx=8)
        self.btn_off = tk.Button(self.motor_row, text="MACHINE OFF", font=("Helvetica", 14, "bold"), bg="#D12D2D", fg="black", padx=18, pady=8, state="disabled", command=self.machine_off)
        self.btn_off.pack(side="left", padx=8)

        # Add user button
        self.add_btn = tk.Button(right, text="➕ ADD USER", font=("Helvetica", 16, "bold"), bg="#13A88E", fg="white", padx=20, pady=10, command=self.on_add_user)
        self.add_btn.pack(pady=(18,6), fill="x")

        self.footer = tk.Label(root, text="Developed by Vadeendra Karanam [CSE-IoT 2026 Batch]", font=("Helvetica", 12), bg="#071428", fg="#9FB9BE")
        self.footer.pack(side="bottom", pady=6)

        # start update loops
        self.update_clock()
        self.update_frame()
        self.update_status_loop()
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    def toggle_fullscreen(self, event=None):
        self.root.attributes("-fullscreen", not self.root.attributes("-fullscreen"))

    def update_clock(self):
        self.time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %I:%M:%S %p"))
        if not self.stop_event.is_set():
            self.root.after(1000, self.update_clock)

    def update_frame(self):
        with frame_mutex:
            local = None if frame is None else frame.copy()
        if local is not None:
            try:
                img = Image.fromarray(local).resize((640, 480))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.configure(image=imgtk)
            except Exception:
                traceback.print_exc()
        if not self.stop_event.is_set():
            self.root.after(30, self.update_frame)

    def update_status_loop(self):
        global current_user, auth_method, machine_state, last_machine_user, last_machine_method
        now = time.time()
        if now - active_rfid["last_seen"] > GUI_DETECT_TIMEOUT:
            active_rfid["text"] = "None"
        if now - active_face["last_seen"] > GUI_DETECT_TIMEOUT:
            active_face["text"] = "None"

        self.rfid_label.config(text=f"RFID Detected : {active_rfid['text']}")
        self.face_label.config(text=f"Face Detected : {active_face['text']}")

        with machine_mutex:
            ms = machine_state
            lm_user = last_machine_user
            lm_method = last_machine_method

        if ms == "ON" and lm_user:
            self.status_label.config(text=f"Status : Machine ON by {lm_user} - {lm_method}", fg="#9BFFB8")
        else:
            self.status_label.config(text="Status : Machine OFF", fg="#FFD27A")

        # Motor GUI coloring and button enable/disable
        with det_mutex:
            user = current_user

        if user is not None:
            # enable controls and set motor frame to green
            self.btn_on.config(state="normal", fg="black")
            self.btn_off.config(state="normal", fg="black")
            self.motor_frame.config(bg="#13A88E")
            self.motor_title.config(bg="#13A88E")
            self.motor_row.config(bg="#13A88E")
            for w in self.motor_frame.winfo_children():
                try:
                    w.config(bg="#13A88E")
                except Exception:
                    pass
        else:
            # black when no detection
            self.btn_on.config(state="disabled", fg="black")
            self.btn_off.config(state="disabled", fg="black")
            self.motor_frame.config(bg="black")
            self.motor_title.config(bg="black")
            self.motor_row.config(bg="black")
            for w in self.motor_frame.winfo_children():
                try:
                    w.config(bg="black")
                except Exception:
                    pass

        if not self.stop_event.is_set():
            self.root.after(300, self.update_status_loop)

    # -------- POPUP WINDOW (6 sec) --------
    def show_popup(self, msg, bg_color):
        try:
            win = tk.Toplevel(self.root)
            win.title("Message")
            # geometry centers roughly; adjust as needed
            win.geometry("480x120+520+300")
            win.configure(bg=bg_color)
            win.attributes("-topmost", True)
            win.resizable(False, False)

            label = tk.Label(
                win,
                text=msg,
                font=("Helvetica", 18, "bold"),
                bg=bg_color,
                fg="white",
                wraplength=440,
                justify="center"
            )
            label.pack(expand=True, fill="both", padx=10, pady=10)

            # Auto-close after 6 seconds
            win.after(6000, win.destroy)
        except Exception:
            # If GUI not fully ready or error occurs, ignore silently
            pass

    # ---------- motor actions ----------
    def machine_on(self):
        global current_user, auth_method, machine_state, last_machine_user, last_machine_method
        with det_mutex:
            user = current_user
            method = auth_method

        if user is None:
            self.status_label.config(text="Status : No authenticated user")
            return

        if GPIO:
            try:
                GPIO.output(MOTOR_GPIO, GPIO.HIGH)
            except Exception as e:
                print("GPIO error on MOTOR ON:", e)

        with machine_mutex:
            machine_state = "ON"
            last_machine_user = user
            last_machine_method = method

        log_machine_access("ON", user, method)
        self.status_label.config(text=f"Machine accessed by {user} - {method} (ON)")

        with det_mutex:
            current_user = None
            auth_method = None

    def machine_off(self):
        global current_user, auth_method, machine_state, last_machine_user, last_machine_method
        with det_mutex:
            user = current_user
            method = auth_method

        if user is None:
            self.status_label.config(text="Status : No authenticated user")
            return

        if GPIO:
            try:
                GPIO.output(MOTOR_GPIO, GPIO.LOW)
            except Exception as e:
                print("GPIO error on MOTOR OFF:", e)

        with machine_mutex:
            machine_state = "OFF"
            last_machine_user = user
            last_machine_method = method

        log_machine_access("OFF", user, method)
        self.status_label.config(text=f"Machine accessed by {user} - {method} (OFF)")

        with det_mutex:
            current_user = None
            auth_method = None

    def on_add_user(self):
        self.stop_event.set()
        if camera_cap:
            camera_cap.release()
        self.root.destroy()
        subprocess.Popen([sys.executable, "/home/project/Desktop/Att/add.py"])
        os._exit(0)

    def on_close(self):
        self.stop_event.set()
        if camera_cap:
            camera_cap.release()
        try:
            if GPIO:
                GPIO.output(MOTOR_GPIO, GPIO.LOW)
        except Exception:
            pass
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
    # global GUI ref so detection threads can show popups (wrapped in try/except)
    global gui_ref
    gui_ref = SentinelGUI(root, stop_event)
    root.mainloop()
    stop_event.set()

if __name__ == "__main__":
    start_all()
