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
from threading import Thread, Lock

# ---------------- GPIO Setup ----------------
RELAY_GPIO = 11
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)

# ---------------- CSV File ----------------
ACCESS_CSV = "access_log.csv"

# ---------------- RFID Setup ----------------
reader = SimpleMFRC522()

# ---------------- Load RFID Tags ----------------
def load_registered_rfid_tags():
    tags = {}
    import sqlite3
    conn = sqlite3.connect("rfid_tags.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS rfid_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rfid_id INTEGER UNIQUE,
            name TEXT
        )
    """)
    cursor.execute("SELECT rfid_id, name FROM rfid_users")
    rows = cursor.fetchall()
    for rfid_id, name in rows:
        tags[int(rfid_id)] = name
    conn.close()
    return tags

rfid_tags = load_registered_rfid_tags()

# ---------------- CSV Logging ----------------
csv_lock = Lock()
def log_access(name, access_type, open_time="", close_time=""):
    with csv_lock:
        file_exists = os.path.exists(ACCESS_CSV)
        with open(ACCESS_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Name", "Type", "Open Timestamp", "Close Timestamp"])
            writer.writerow([name, access_type, open_time, close_time])

# ---------------- Face Recognition Setup ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# ---------------- Access System ----------------
class AccessSystem:
    def __init__(self):
        # Face DB
        self.face_features_known_list = []
        self.face_name_known_list = []
        self.load_face_database()

        # Lock state
        self.lock_open = False
        self.current_user = None
        self.lock_open_start_time = 0
        self.lock_duration = 10       # Lock stays open 15s
        self.cooldown_seconds = 10    # Wait 10s after closing
        self.last_close_time = 0

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Thread-safe lock
        self.state_lock = Lock()

    def load_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                features = [csv_rd.iloc[i][j] for j in range(1, 129)]
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_features_known_list.append(features)

    def euclidean_distance(self, f1, f2):
        return np.linalg.norm(np.array(f1) - np.array(f2))

    # ---------- Face Processing ----------
    def process_face(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue

            faces = detector(frame, 0)
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            detected_names = []

            for face in faces:
                shape = predictor(frame, face)
                face_feature = face_reco_model.compute_face_descriptor(frame, shape)
                distances = [self.euclidean_distance(face_feature, f) for f in self.face_features_known_list]

                if distances and min(distances) < 0.6:
                    name = self.face_name_known_list[distances.index(min(distances))]
                    detected_names.append(name)

                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
                    cv2.putText(frame, name, (face.left(), face.top()-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                    with self.state_lock:
                        # Open only if closed + cooldown passed
                        if (not self.lock_open) and (time.time() - self.last_close_time >= self.cooldown_seconds):
                            GPIO.output(RELAY_GPIO, GPIO.HIGH)
                            self.lock_open = True
                            self.current_user = name
                            self.lock_open_start_time = time.time()
                            print(f"🔓 Lock opened by {name} (Face)")
                            log_access(name, "Face", open_time=now_str)

                        # If lock already open by another user → just log
                        elif self.lock_open and name != self.current_user:
                            log_access(name, "Face", open_time=now_str)

                else:
                    # Unknown face → log only
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,0,255),2)
                    cv2.putText(frame, "Unknown", (face.left(), face.top()-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),1)
                    log_access("Unknown", "Face", open_time=now_str)

            # Automatic lock close after 15s if opener still present
            with self.state_lock:
                if self.lock_open and time.time() - self.lock_open_start_time >= self.lock_duration:
                    if self.current_user in detected_names:
                        GPIO.output(RELAY_GPIO, GPIO.LOW)
                        now_close_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"🔒 Lock closed by {self.current_user}")
                        log_access(self.current_user, "Face", close_time=now_close_str)
                        self.lock_open = False
                        self.current_user = None
                        self.last_close_time = time.time()
                    else:
                        # Reset timer if opener not seen
                        self.lock_open_start_time = time.time()

            cv2.imshow("Access System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

    # ---------- RFID Processing ----------
    def process_rfid(self):
        while True:
            try:
                rfid_id, text = reader.read_no_block()
            except:
                rfid_id = None

            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if rfid_id:
                with self.state_lock:
                    if rfid_id in rfid_tags:
                        tag_name = rfid_tags[rfid_id]

                        # Open only if closed + cooldown passed
                        if (not self.lock_open) and (time.time() - self.last_close_time >= self.cooldown_seconds):
                            GPIO.output(RELAY_GPIO, GPIO.HIGH)
                            self.lock_open = True
                            self.current_user = tag_name
                            self.lock_open_start_time = time.time()
                            print(f"🔓 Lock opened by {tag_name} (RFID)")
                            log_access(tag_name, "RFID", open_time=now_str)

                        # If lock already open by another user → just log
                        elif self.lock_open and tag_name != self.current_user:
                            log_access(tag_name, "RFID", open_time=now_str)

                    else:
                        # Unknown RFID → log only
                        print(f"❌ Unknown RFID tag: {rfid_id}")
                        log_access("Unknown", "RFID", open_time=now_str)

            # Automatic close (same logic as face)
            with self.state_lock:
                if self.lock_open and time.time() - self.lock_open_start_time >= self.lock_duration:
                    if self.current_user == rfid_tags.get(rfid_id, None):
                        GPIO.output(RELAY_GPIO, GPIO.LOW)
                        now_close_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"🔒 Lock closed by {self.current_user}")
                        log_access(self.current_user, "RFID", close_time=now_close_str)
                        self.lock_open = False
                        self.current_user = None
                        self.last_close_time = time.time()
                    else:
                        # Reset timer if opener not seen
                        self.lock_open_start_time = time.time()

            time.sleep(0.1)

    # ---------- Run System ----------
    def run(self):
        t1 = Thread(target=self.process_face)
        t2 = Thread(target=self.process_rfid)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

if __name__ == "__main__":
    print("🔑 Access System Running...")
    system = AccessSystem()
    system.run()
