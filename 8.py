import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib")import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import dlib
import numpy as np
import cv2
import pandas as pd
import time
import datetime
import csv
import sqlite3
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
from threading import Lock, Thread

# ---------------- GPIO SETUP ----------------
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)  # initially closed

# ---------------- LOCK STATE FILE ----------------
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
            f.flush()
            os.fsync(f.fileno())

# ---------------- CSV LOGGING ----------------
def log_face(name, open_time="", close_time=""):
    CSV_FILE = "face_access_log.csv"
    rows = []
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
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
    pd.DataFrame(rows).to_csv(CSV_FILE, index=False)

def log_rfid(uid, tag_name, open_time="", close_time=""):
    CSV_FILE = "rfid_access_log.csv"
    rows = []
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        rows = df.to_dict("records")
    updated = False
    for row in rows:
        if row["UID"] == str(uid) and row["Close Timestamp"] == "":
            if close_time:
                row["Close Timestamp"] = close_time
            updated = True
            break
    if not updated:
        rows.append({"UID": str(uid), "Tag Name": tag_name, "Open Timestamp": open_time, "Close Timestamp": close_time})
    pd.DataFrame(rows).to_csv(CSV_FILE, index=False)

# ---------------- DLIB SETUP ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# ---------------- FACE ACCESS SYSTEM ----------------
class FaceAccessSystem(Thread):
    def __init__(self):
        super().__init__()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.load_face_database()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.current_user = None
        self.lock_open_time = 0
        self.ignore_duration = 10  # seconds before re-detect

    def load_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                features = [csv_rd.iloc[i][j] for j in range(1, 129)]
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_features_known_list.append(features)
        else:
            print("❌ Face database not found.")
            exit()

    def euclidean_distance(self, f1, f2):
        return np.linalg.norm(np.array(f1) - np.array(f2))

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            faces = detector(frame, 0)
            detected_names = []

            for face in faces:
                shape = predictor(frame, face)
                feature = face_reco_model.compute_face_descriptor(frame, shape)
                distances = [self.euclidean_distance(feature, f) for f in self.face_features_known_list]
                if distances and min(distances) < 0.6:
                    name = self.face_name_known_list[distances.index(min(distances))]
                    detected_names.append(name)
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
                    cv2.putText(frame, name, (face.left(), face.top()-10), self.font, 0.6, (0,255,255),1)

                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    state = get_lock_state()
                    lock_status = state.get("status", "CLOSED")
                    opener = state.get("system", "")

                    # Only act if lock closed OR re-detect after ignore duration
                    if lock_status == "CLOSED" or (self.current_user == name and time.time() - self.lock_open_time > self.ignore_duration):
                        if lock_status == "CLOSED":
                            GPIO.output(RELAY_GPIO, GPIO.HIGH)
                            self.current_user = name
                            self.lock_open_time = time.time()
                            set_lock_state("OPEN", f"FACE:{name}")
                            log_face(name, open_time=now)
                            print(f"🔓 Lock opened by {name}")
                        elif self.current_user == name:
                            GPIO.output(RELAY_GPIO, GPIO.LOW)
                            close_time = now
                            set_lock_state("CLOSED", "NONE")
                            log_face(name, close_time=close_time)
                            print(f"🔒 Lock closed by {name}")
                            self.current_user = None

                else:
                    # Unknown face
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,0,255), 2)
                    cv2.putText(frame, "Unknown", (face.left(), face.top()-10), self.font, 0.6, (0,0,255),1)
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"❌ Unknown face detected at {now}")
                    log_face("Unknown", open_time=now)

            cv2.imshow("Face Access", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.05)
        self.cap.release()
        cv2.destroyAllWindows()

# ---------------- RFID ACCESS SYSTEM ----------------
class RFIDAccessSystem(Thread):
    def __init__(self):
        super().__init__()
        self.reader = SimpleMFRC522()
        self.rfid_tags = self.load_registered_rfid_tags()
        self.current_access_tag = None
        self.ignore_duration = 10
        self.last_detect_time = 0
        self.other_tags_detected = set()

    def load_registered_rfid_tags(self):
        tags = {}
        conn = sqlite3.connect("rfid_tags.db")
        cursor = conn.cursor()
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS rfid_users (id INTEGER PRIMARY KEY AUTOINCREMENT, rfid_id INTEGER UNIQUE, name TEXT)"
        )
        cursor.execute("SELECT rfid_id, name FROM rfid_users")
        rows = cursor.fetchall()
        for rfid_id, name in rows:
            tags[int(rfid_id)] = name
        conn.close()
        return tags

    def run(self):
        while True:
            rfid_id, text = self.reader.read_no_block()
            if rfid_id:
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                state = get_lock_state()
                lock_status = state.get("status", "CLOSED")
                opener = state.get("system", "")

                if rfid_id in self.rfid_tags:
                    tag_name = self.rfid_tags[rfid_id]
                    if lock_status == "CLOSED" or (self.current_access_tag == rfid_id and time.time() - self.last_detect_time > self.ignore_duration):
                        if lock_status == "CLOSED":
                            GPIO.output(RELAY_GPIO, GPIO.HIGH)
                            self.current_access_tag = rfid_id
                            self.last_detect_time = time.time()
                            set_lock_state("OPEN", f"RFID:{tag_name}")
                            log_rfid(rfid_id, tag_name, open_time=now)
                            print(f"🔓 Lock opened by {tag_name}")
                        elif self.current_access_tag == rfid_id:
                            GPIO.output(RELAY_GPIO, GPIO.LOW)
                            close_time = now
                            set_lock_state("CLOSED", "NONE")
                            log_rfid(rfid_id, tag_name, close_time=close_time)
                            print(f"🔒 Lock closed by {tag_name}")
                            self.current_access_tag = None
                    else:
                        # Other tag while lock is open → log only
                        if rfid_id not in self.other_tags_detected:
                            log_rfid(rfid_id, tag_name, open_time=now)
                            self.other_tags_detected.add(rfid_id)
                            print(f"❌ Other RFID detected: {tag_name} at {now}")
                else:
                    print(f"❌ Unknown RFID tag: {rfid_id}")
            time.sleep(0.1)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        face_system = FaceAccessSystem()
        rfid_system = RFIDAccessSystem()
        face_system.start()
        rfid_system.start()
        face_system.join()
        rfid_system.join()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        set_lock_state("CLOSED", "NONE")
        # Do not cleanup GPIO in case other scripts are running


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
