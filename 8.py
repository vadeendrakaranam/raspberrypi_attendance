import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import dlib
import numpy as np
import cv2
import pandas as pd
import time
import datetime
import csv
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import sqlite3
from threading import Thread, Lock

# ---------------- GPIO Setup ----------------
RELAY_GPIO = 11
RELAY_ON = GPIO.HIGH
RELAY_OFF = GPIO.LOW
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, RELAY_OFF)

# ---------------- CSV & Lock State ----------------
FACE_CSV = "face_access_log.csv"
RFID_CSV = "rfid_access_log.csv"
LOCK_STATE_FILE = "lock_state.txt"
lock_file_mutex = Lock()  # Thread-safe lock for lock_state.txt

def get_lock_state():
    with lock_file_mutex:
        if not os.path.exists(LOCK_STATE_FILE):
            return {"status": "CLOSED", "system": None}
        with open(LOCK_STATE_FILE, "r") as f:
            lines = f.readlines()
        state = {}
        for line in lines:
            if "=" in line:
                k, v = line.strip().split("=")
                state[k] = v
        return state

def set_lock_state(status, system):
    with lock_file_mutex:
        with open(LOCK_STATE_FILE, "w") as f:
            f.write(f"status={status}\n")
            f.write(f"system={system}\n")
            f.write(f"timestamp={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ---------------- Face Logging ----------------
def log_face_open(name, open_time):
    file_exists = os.path.exists(FACE_CSV)
    with open(FACE_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Open Timestamp", "Close Timestamp"])
        writer.writerow([name, open_time, ""])

def log_face_close(name, close_time):
    if not os.path.exists(FACE_CSV):
        return
    df = pd.read_csv(FACE_CSV)
    mask = (df["Name"] == name) & (df["Close Timestamp"] == "")
    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "Close Timestamp"] = close_time
        df.to_csv(FACE_CSV, index=False)

# ---------------- RFID Logging ----------------
def read_csv(csv_file):
    if not os.path.exists(csv_file):
        return []
    with open(csv_file, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv(csv_file, rows):
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["UID", "Tag Name", "Open Timestamp", "Close Timestamp"])
        writer.writeheader()
        writer.writerows(rows)

def log_rfid_access(uid, tag_name, open_time="", close_time=""):
    rows = read_csv(RFID_CSV)
    updated = False
    for row in rows:
        if row["UID"] == str(uid) and row["Close Timestamp"] == "":
            if close_time:
                row["Close Timestamp"] = close_time
            updated = True
            break
    if not updated:
        rows.append({
            "UID": str(uid),
            "Tag Name": tag_name,
            "Open Timestamp": open_time,
            "Close Timestamp": close_time
        })
    write_csv(RFID_CSV, rows)

# ---------------- Load Face Database ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

face_name_known_list = []
face_features_known_list = []

if os.path.exists("data/features_all.csv"):
    csv_rd = pd.read_csv("data/features_all.csv", header=None)
    for i in range(csv_rd.shape[0]):
        features = [csv_rd.iloc[i][j] for j in range(1, 129)]
        face_name_known_list.append(csv_rd.iloc[i][0])
        face_features_known_list.append(features)
else:
    print("❌ Face database not found.")
    exit()

def euclidean_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

# ---------------- Load RFID Tags ----------------
def load_registered_rfid_tags():
    tags = {}
    conn = sqlite3.connect("rfid_tags.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS rfid_users (id INTEGER PRIMARY KEY AUTOINCREMENT, rfid_id INTEGER UNIQUE, name TEXT)")
    cursor.execute("SELECT rfid_id, name FROM rfid_users")
    rows = cursor.fetchall()
    for rfid_id, name in rows:
        tags[int(rfid_id)] = name
    conn.close()
    return tags

rfid_tags = load_registered_rfid_tags()
reader = SimpleMFRC522()

# ---------------- Shared Variables ----------------
lock_open = False
current_user = None
lock_open_start_time = 0
lock_duration = 15
other_faces = {}
current_access_tag = None
other_tags_detected = set()

# ---------------- Face Detection Thread ----------------
def face_thread():
    global lock_open, current_user, lock_open_start_time, other_faces
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector(frame, 0)
        detected_names = []

        for face in faces:
            shape = predictor(frame, face)
            face_feature = face_reco_model.compute_face_descriptor(frame, shape)
            distances = [euclidean_distance(face_feature, f) for f in face_features_known_list]

            if distances and min(distances) < 0.6:
                name = face_name_known_list[distances.index(min(distances))]
                detected_names.append(name)
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255,255,255), 2)
                cv2.putText(frame, name, (face.left(), face.top()-10), font, 0.6, (0,255,255),1)
                now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                state = get_lock_state()

                if state["status"] == "CLOSED":
                    # Lock free → open it
                    GPIO.output(RELAY_GPIO, RELAY_ON)
                    set_lock_state("OPEN", "FACE")
                    lock_open = True
                    current_user = name
                    lock_open_start_time = time.time()
                    print(f"🔓 Lock opened by {name}")
                    log_face_open(name, now_time_str)
                else:
                    # Lock already open → log if other face
                    if name != state["system"] and name not in other_faces:
                        print(f"❌ Other detected: {name} at {now_time_str}")
                        log_face_open(name, now_time_str)
                        other_faces[name] = now_time_str

        # Auto-close only for the same user
        if lock_open and current_user in detected_names:
            if time.time() - lock_open_start_time >= lock_duration:
                GPIO.output(RELAY_GPIO, RELAY_OFF)
                set_lock_state("CLOSED", "NONE")
                close_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"🔒 Lock closed by {current_user}")
                log_face_close(current_user, close_time_str)
                lock_open = False
                current_user = None
                other_faces.clear()

        cv2.imshow("Access System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ---------------- RFID Thread ----------------
def rfid_thread():
    global lock_open, current_access_tag, other_tags_detected, lock_open_start_time
    while True:
        rfid_id, text = reader.read_no_block()
        if rfid_id:
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if rfid_id in rfid_tags:
                tag_name = rfid_tags[rfid_id]
                state = get_lock_state()

                if state["status"] == "CLOSED":
                    # Lock free → open it
                    GPIO.output(RELAY_GPIO, RELAY_ON)
                    set_lock_state("OPEN", "RFID")
                    current_access_tag = rfid_id
                    lock_open_start_time = time.time()
                    print(f"🔓 Lock opened by {tag_name} at {now}")
                    log_rfid_access(rfid_id, tag_name, open_time=now)
                else:
                    # Lock open → log other tag
                    if rfid_id != int(state["system"]) and rfid_id not in other_tags_detected:
                        print(f"❌ Other tag detected: {tag_name} at {now}")
                        log_rfid_access(rfid_id, tag_name, open_time=now)
                        other_tags_detected.add(rfid_id)

        # Auto-close for RFID (optional, can be same as face)
        if lock_open and current_access_tag == rfid_id:
            if time.time() - lock_open_start_time >= lock_duration:
                GPIO.output(RELAY_GPIO, RELAY_OFF)
                set_lock_state("CLOSED", "NONE")
                close_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"🔒 Lock closed by {tag_name}")
                log_rfid_access(rfid_id, tag_name, close_time=close_time_str)
                lock_open = False
                current_access_tag = None
                other_tags_detected.clear()

        time.sleep(0.05)

# ---------------- Run Threads ----------------
try:
    Thread(target=face_thread, daemon=True).start()
    Thread(target=rfid_thread, daemon=True).start()
    print("🔑 Combined Face + RFID System Running (Threaded)...")
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    GPIO.output(RELAY_GPIO, RELAY_OFF)
    set_lock_state("CLOSED", "NONE")
    GPIO.cleanup()
