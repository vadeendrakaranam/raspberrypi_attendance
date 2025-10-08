import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import tkinter as tk
from PIL import Image, ImageTk
import threading
import time
import datetime
import csv
import os
import cv2
import RPi.GPIO as GPIO
import dlib
import pandas as pd
from mfrc522 import SimpleMFRC522

# ---------------- GPIO SETUP ----------------
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)

# ---------------- GLOBAL STATE ----------------
auth_state = {"face_verified": False, "rfid_verified": False, "name": None}

FACE_CSV = "face_access_log.csv"
RFID_CSV = "rfid_access_log.csv"

# ---------------- LOGGING ----------------
def log_event(csv_file, name, event):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["Name","Event","Timestamp"])
        writer.writerow([name,event,now])

# ---------------- LOAD FACE DATABASE ----------------
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

face_names = []
face_features = []

if os.path.exists("data/features_all.csv"):
    df = pd.read_csv("data/features_all.csv", header=None)
    for i in range(df.shape[0]):
        face_names.append(df.iloc[i, 0])
        face_features.append(list(df.iloc[i, 1:129]))
else:
    print("❌ Face database not found.")
    exit()

# ---------------- FACE DETECTION ----------------
def detect_face(face_label, status_label, tick_label):
    cap = cv2.VideoCapture(0)
    detected = False

    while not detected:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray)

        for face in faces:
            shape = predictor(frame, face)
            feature = face_rec_model.compute_face_descriptor(frame, shape)
            distances = [np.linalg.norm(np.array(feature)-np.array(f)) for f in face_features]

            if distances and min(distances) < 0.6:
                name = face_names[distances.index(min(distances))]
                detected = True
                auth_state["face_verified"] = True
                auth_state["name"] = name
                status_label.config(text=f"✅ Face Verified: {name}")
                tick_label.config(text="✅")
                log_event(FACE_CSV, name, "Face Detected")
            else:
                status_label.config(text="❌ Unknown Face")

            # Draw rectangle
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        # Convert for Tkinter
        label_width = face_label.winfo_width() or 400
        label_height = face_label.winfo_height() or 300
        frame_resized = cv2.resize(frame, (label_width, label_height))
        img = Image.fromarray(cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        face_label.imgtk = imgtk
        face_label.configure(image=imgtk)
        face_label.update()
        time.sleep(0.03)

    cap.release()

# ---------------- RFID DETECTION ----------------
def detect_rfid(status_label, tick_label):
    status_label.config(text="Reading...")
    reader = SimpleMFRC522()
    rfid_detected = False

    while not rfid_detected:
        rfid_id, text = reader.read_no_block()
        if rfid_id:
            tag_name = text.strip() if text else str(rfid_id)
            if auth_state["face_verified"] and auth_state["name"] == tag_name:
                auth_state["rfid_verified"] = True
                status_label.config(text=f"✅ RFID Verified ({tag_name})")
                tick_label.config(text="✅")
                log_event(RFID_CSV, tag_name, "RFID Detected")
            else:
                status_label.config(text="❌ RFID mismatch with Face")
            rfid_detected = True
        time.sleep(0.1)

# ---------------- RELAY CONTROL ----------------
def open_relay_control(root_window):
    if auth_state["face_verified"] and auth_state["rfid_verified"]:
        relay_win = tk.Toplevel()
        relay_win.title("Relay Control Panel")
        relay_win.geometry("300x200")
        relay_win.grab_set()

        def relay_on():
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            print("Relay ON")

        def relay_off():
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            print("Relay OFF")

        tk.Label(relay_win, text=f"Welcome {auth_state['name']}", font=("Arial",12,"bold")).pack(pady=10)
        tk.Button(relay_win, text="Relay ON", command=relay_on, width=15).pack(pady=5)
        tk.Button(relay_win, text="Relay OFF", command=relay_off, width=15).pack(pady=5)

        def on_close():
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            auth_state["face_verified"] = False
            auth_state["rfid_verified"] = False
            auth_state["name"] = None
            relay_win.destroy()

        relay_win.protocol("WM_DELETE_WINDOW", on_close)
        relay_win.mainloop()
    else:
        print("Both Face and RFID must be verified first.")

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Access Control System")
root.geometry("700x450")

# Navbar with date & time
navbar = tk.Label(root, text="", bg="lightgrey", font=("Arial", 12))
navbar.pack(fill=tk.X)
def update_time():
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    navbar.config(text=f"Current Date & Time: {now}")
    root.after(500, update_time)
update_time()

# Main frame with 2 sections
main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Left Frame: Face Detection
left_frame = tk.LabelFrame(main_frame, text="Face Detection", width=320, height=300)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
face_label = tk.Label(left_frame, bg="black")
face_label.pack(pady=5, fill=tk.BOTH, expand=True)
face_status_label = tk.Label(left_frame, text="Not Started")
face_status_label.pack(pady=5)
face_tick_label = tk.Label(left_frame, text="")
face_tick_label.pack()
tk.Button(left_frame, text="Start Face Detection", 
          command=lambda: threading.Thread(target=detect_face, args=(face_label, face_status_label, face_tick_label)).start()).pack(pady=5)

# Right Frame: RFID Detection
right_frame = tk.LabelFrame(main_frame, text="RFID Detection", width=320, height=300)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
rfid_status_label = tk.Label(right_frame, text="Not Started")
rfid_status_label.pack(pady=20)
rfid_tick_label = tk.Label(right_frame, text="")
rfid_tick_label.pack()
tk.Button(right_frame, text="Start RFID Detection", 
          command=lambda: threading.Thread(target=detect_rfid, args=(rfid_status_label, rfid_tick_label)).start()).pack(pady=5)

# Open Relay Control button
tk.Button(root, text="Open Relay Control", command=lambda: open_relay_control(root), bg="green", fg="white").pack(pady=10)

root.mainloop()

GPIO.output(RELAY_GPIO, GPIO.LOW)
GPIO.cleanup()
