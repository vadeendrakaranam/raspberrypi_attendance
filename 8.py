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
from threading import Lock, Thread

# --- GPIO setup ---
RELAY_GPIO = 11
RELAY_ON = GPIO.HIGH
RELAY_OFF = GPIO.LOW
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, RELAY_OFF)

# --- Lock state ---
LOCK_STATE_FILE = "lock_state.txt"
lock_file_mutex = Lock()

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

# --- CSV Logging ---
FACE_CSV = "face_access_log.csv"
RFID_CSV = "rfid_access_log.csv"

def log_access(csv_file, name, open_time="", close_time=""):
    if not os.path.exists(csv_file):
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Open Timestamp", "Close Timestamp"])
    rows = []
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        rows = df.to_dict("records")
    updated = False
    for row in rows:
        if row["Name"] == name and row["Close Timestamp"] == "":
            if close_time:
                row["Close Timestamp"] = close_time
            updated = True
            break
    if not updated:
        rows.append({"Name": name, "Open Timestamp": open_time, "Close Timestamp": close_time})
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)

# --- Load registered RFID tags ---
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

# --- Face recognition setup ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

def load_face_database():
    face_features_known_list = []
    face_name_known_list = []
    if os.path.exists("data/features_all.csv"):
        csv_rd = pd.read_csv("data/features_all.csv", header=None)
        for i in range(csv_rd.shape[0]):
            features = [csv_rd.iloc[i][j] for j in range(1, 129)]
            face_name_known_list.append(csv_rd.iloc[i][0])
            face_features_known_list.append(features)
    return face_features_known_list, face_name_known_list

face_features_known_list, face_name_known_list = load_face_database()

def euclidean_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

# --- Shared lock variables ---
current_user = None
lock_open_start_time = 0
lock_duration = 15
other_detections = set()
lock_mutex = Lock()

# --- Face recognition thread ---
def face_thread():
    global current_user, lock_open_start_time
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector(frame, 0)
        detected_names = []
        for face in faces:
            shape = predictor(frame, face)
            feature = face_reco_model.compute_face_descriptor(frame, shape)
            distances = [euclidean_distance(feature, f) for f in face_features_known_list]
            if distances and min(distances) < 0.6:
                name = face_name_known_list[distances.index(min(distances))]
                detected_names.append(name)
                now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                with lock_mutex:
                    state = get_lock_state()
                    if state["status"] == "CLOSED":
                        GPIO.output(RELAY_GPIO, RELAY_ON)
                        current_user = f"FACE:{name}"
                        lock_open_start_time = time.time()
                        set_lock_state("OPEN", current_user)
                        print(f"🔓 Lock opened by FACE:{name}")
                        log_access(FACE_CSV, name, open_time=now_time_str)
                    elif state["status"] == "OPEN" and state["system"] != f"FACE:{name}":
                        if name not in other_detections:
                            print(f"Other FACE detected: {name} at {now_time_str}")
                            log_access(FACE_CSV, name, open_time=now_time_str)
                            other_detections.add(name)
        # Automatic close
        with lock_mutex:
            if current_user and current_user.startswith("FACE:"):
                elapsed = time.time() - lock_open_start_time
                if elapsed >= lock_duration and any(f"FACE:{n}" in detected_names for n in face_name_known_list):
                    GPIO.output(RELAY_GPIO, RELAY_OFF)
                    close_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"🔒 Lock closed by {current_user}")
                    log_access(FACE_CSV, current_user.split("FACE:")[1], close_time=close_time_str)
                    set_lock_state("CLOSED", "NONE")
                    current_user = None
                    other_detections.clear()
        cv2.imshow("Face Access", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)
    cap.release()
    cv2.destroyAllWindows()

# --- RFID thread ---
def rfid_thread():
    global current_user, lock_open_start_time
    while True:
        rfid_id, _ = reader.read_no_block()
        if rfid_id:
            now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if rfid_id in rfid_tags:
                name = rfid_tags[rfid_id]
                with lock_mutex:
                    state = get_lock_state()
                    if state["status"] == "CLOSED":
                        GPIO.output(RELAY_GPIO, RELAY_ON)
                        current_user = f"RFID:{name}"
                        lock_open_start_time = time.time()
                        set_lock_state("OPEN", current_user)
                        print(f"🔓 Lock opened by RFID:{name}")
                        log_access(RFID_CSV, name, open_time=now_time_str)
                    elif state["status"] == "OPEN" and state["system"] != f"RFID:{name}":
                        if name not in other_detections:
                            print(f"Other RFID detected: {name} at {now_time_str}")
                            log_access(RFID_CSV, name, open_time=now_time_str)
                            other_detections.add(name)
        time.sleep(0.5)
        # Auto-close
        with lock_mutex:
            if current_user and current_user.startswith("RFID:"):
                elapsed = time.time() - lock_open_start_time
                if elapsed >= lock_duration:
                    GPIO.output(RELAY_GPIO, RELAY_OFF)
                    close_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"🔒 Lock closed by {current_user}")
                    log_access(RFID_CSV, current_user.split("RFID:")[1], close_time=close_time_str)
                    set_lock_state("CLOSED", "NONE")
                    current_user = None
                    other_detections.clear()

# --- Main ---
if __name__ == "__main__":
    try:
        t1 = Thread(target=face_thread, daemon=True)
        t2 = Thread(target=rfid_thread, daemon=True)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.output(RELAY_GPIO, RELAY_OFF)
        set_lock_state("CLOSED", "NONE")
        GPIO.cleanup()
