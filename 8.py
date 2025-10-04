import sys, os
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
FACE_CSV = "face_access_log.csv"
RFID_CSV = "rfid_access_log.csv"

def log_face(name, open_time="", close_time=""):
    rows = []
    if os.path.exists(FACE_CSV):
        df = pd.read_csv(FACE_CSV)
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
    pd.DataFrame(rows).to_csv(FACE_CSV, index=False)

def log_rfid(uid, tag_name, open_time="", close_time=""):
    rows = []
    if os.path.exists(RFID_CSV):
        df = pd.read_csv(RFID_CSV)
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
    pd.DataFrame(rows).to_csv(RFID_CSV, index=False)

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
        self.ignore_duration = 10  # seconds

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

                    if lock_status == "CLOSED":
                        GPIO.output(RELAY_GPIO, GPIO.HIGH)
                        self.current_user = name
                        self.lock_open_time = time.time()
                        set_lock_state("OPEN", f"FACE:{name}")
                        log_face(name, open_time=now)
                        print(f"🔓 Lock opened by {name}")
                    elif self.current_user == name and time.time() - self.lock_open_time > self.ignore_duration:
                        GPIO.output(RELAY_GPIO, GPIO.LOW)
                        close_time = now
                        set_lock_state("CLOSED", "NONE")
                        log_face(name, close_time=close_time)
                        print(f"🔒 Lock closed by {name}")
                        self.current_user = None
                    else:
                        # Other faces logged but don't affect lock
                        if name != self.current_user:
                            log_face(name, open_time=now)
                            print(f"❌ Other face detected: {name} at {now}")
                else:
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,0,255), 2)
                    cv2.putText(frame, "Unknown", (face.left(), face.top()-10), self.font, 0.6, (0,0,255),1)
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_face("Unknown", open_time=now)
                    print(f"❌ Unknown face detected at {now}")

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
                    if lock_status == "CLOSED":
                        GPIO.output(RELAY_GPIO, GPIO.HIGH)
                        self.current_access_tag = rfid_id
                        self.last_detect_time = time.time()
                        set_lock_state("OPEN", f"RFID:{tag_name}")
                        log_rfid(rfid_id, tag_name, open_time=now)
                        print(f"🔓 Lock opened by {tag_name}")
                    elif self.current_access_tag == rfid_id and time.time() - self.last_detect_time > self.ignore_duration:
                        GPIO.output(RELAY_GPIO, GPIO.LOW)
                        close_time = now
                        set_lock_state("CLOSED", "NONE")
                        log_rfid(rfid_id, tag_name, close_time=close_time)
                        print(f"🔒 Lock closed by {tag_name}")
                        self.current_access_tag = None
                    else:
                        # Other tags logged
                        if rfid_id != self.current_access_tag and rfid_id not in self.other_tags_detected:
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
