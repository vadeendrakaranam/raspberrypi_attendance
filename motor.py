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

# ---------------- GPIO SETUP ----------------
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)  # initially off

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

# ---------------- FACE DETECTION ----------------
def detect_face(face_label, status_label, tick_label):
    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    detected = False

    while not detected:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        # Convert image to Tkinter format
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        face_label.imgtk = imgtk
        face_label.configure(image=imgtk)

        if len(faces) > 0:
            name = "Vadeendra"  # Map detected face name here
            auth_state["face_verified"] = True
            auth_state["name"] = name
            status_label.config(text=f"✅ Face Verified: {name}")
            tick_label.config(text="✅")  # Show tick
            log_event(FACE_CSV, name, "Face Detected")
            detected = True

        face_label.update()
        status_label.update()
        tick_label.update()
        time.sleep(0.03)

    cap.release()

# ---------------- RFID DETECTION (SIMULATED) ----------------
def detect_rfid(status_label, tick_label):
    time.sleep(2)
    tag_name = "Vadeendra"  # Simulated RFID name
    if auth_state["face_verified"] and auth_state["name"] == tag_name:
        auth_state["rfid_verified"] = True
        status_label.config(text=f"✅ RFID Verified ({tag_name})")
        tick_label.config(text="✅")  # Show tick
        log_event(RFID_CSV, tag_name, "RFID Detected")
    else:
        status_label.config(text="❌ RFID mismatch with Face")
        tick_label.config(text="")

# ---------------- RELAY CONTROL ----------------
def open_relay_control(face_tick_label, rfid_tick_label, face_status_label, rfid_status_label):
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
            face_tick_label.config(text="")
            rfid_tick_label.config(text="")
            face_status_label.config(text="Not Started")
            rfid_status_label.config(text="Not Started")
            relay_win.destroy()

        relay_win.protocol("WM_DELETE_WINDOW", on_close)
        relay_win.mainloop()
    else:
        print("Both Face and RFID must be verified first.")

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Access Control System")
root.geometry("650x400")

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
left_frame = tk.LabelFrame(main_frame, text="Face Detection", width=300, height=300)
left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

face_label = tk.Label(left_frame, text="Camera Feed", width=30, height=12, bg="black")
face_label.pack(pady=5)

face_status_frame = tk.Frame(left_frame)
face_status_frame.pack()
face_status_label = tk.Label(face_status_frame, text="Not Started")
face_status_label.pack(side=tk.LEFT)
face_tick_label = tk.Label(face_status_frame, text="", fg="green", font=("Arial", 14, "bold"))
face_tick_label.pack(side=tk.LEFT, padx=5)

tk.Button(left_frame, text="Start Face Detection", 
          command=lambda: threading.Thread(target=detect_face, args=(face_label, face_status_label, face_tick_label)).start()).pack(pady=5)

# Right Frame: RFID Detection
right_frame = tk.LabelFrame(main_frame, text="RFID Detection", width=300, height=300)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

rfid_status_frame = tk.Frame(right_frame)
rfid_status_frame.pack(pady=20)
rfid_status_label = tk.Label(rfid_status_frame, text="Not Started")
rfid_status_label.pack(side=tk.LEFT)
rfid_tick_label = tk.Label(rfid_status_frame, text="", fg="green", font=("Arial", 14, "bold"))
rfid_tick_label.pack(side=tk.LEFT, padx=5)

tk.Button(right_frame, text="Start RFID Detection", 
          command=lambda: threading.Thread(target=detect_rfid, args=(rfid_status_label, rfid_tick_label)).start()).pack(pady=5)

# Open Relay Control Button
tk.Button(root, text="Open Relay Control", 
          command=lambda: open_relay_control(face_tick_label, rfid_tick_label, face_status_label, rfid_status_label),
          bg="green", fg="white").pack(pady=10)

root.mainloop()

# Cleanup GPIO on exit
GPIO.output(RELAY_GPIO, GPIO.LOW)
GPIO.cleanup()
