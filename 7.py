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
import sqlite3
from mfrc522 import SimpleMFRC522

# --- GPIO setup ---
RELAY_GPIO = 11
RELAY_ON = GPIO.HIGH
RELAY_OFF = GPIO.LOW
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, RELAY_OFF)

# --- RFID setup ---
reader = SimpleMFRC522()

# --- Load registered RFID tags ---
def load_registered_rfid_tags():
    tags = {}
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

# --- CSV helpers ---
FACE_CSV = "face_access_log.csv"
RFID_CSV = "rfid_access_log.csv"

def log_open(csv_file, identifier, open_time):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Identifier", "Open Timestamp", "Close Timestamp"])
        writer.writerow([identifier, open_time, ""])

def log_close(csv_file, identifier, close_time):
    if not os.path.exists(csv_file):
        return
    df = pd.read_csv(csv_file)
    mask = (df["Identifier"] == identifier) & (df["Close Timestamp"] == "")
    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "Close Timestamp"] = close_time
        df.to_csv(csv_file, index=False)

# --- Face system setup ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class AccessSystem:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.face_features_known_list = []
        self.face_name_known_list = []

        self.lock_open = False
        self.current_user = None
        self.open_time_str = None
        self.lock_start_time = 0
        self.lock_duration = 15      # Lock stays open for 15s
        self.cooldown_seconds = 10   # Cooldown after closing
        self.last_close_time = 0

        self.other_faces = set()
        self.other_tags = set()

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        if not self.load_face_database():
            print("❌ Face database not found.")
            exit()

    def load_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                features = [csv_rd.iloc[i][j] for j in range(1, 129)]
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_features_known_list.append(features)
            return True
        return False

    def euclidean_distance(self, f1, f2):
        return np.linalg.norm(np.array(f1) - np.array(f2))

    def detect_faces(self, frame):
        faces = detector(frame, 0)
        detected_names = []
        for face in faces:
            shape = predictor(frame, face)
            face_feature = face_reco_model.compute_face_descriptor(frame, shape)
            distances = [self.euclidean_distance(face_feature, f) for f in self.face_features_known_list]
            if distances and min(distances) < 0.6:
                name = self.face_name_known_list[distances.index(min(distances))]
                detected_names.append(name)
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
                cv2.putText(frame, name, (face.left(), face.top() - 10), self.font, 0.6, (0, 255, 255), 1)
            else:
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (face.left(), face.top() - 10), self.font, 0.6, (0, 0, 255), 1)
        return detected_names

    def detect_rfid(self):
        rfid_id, text = reader.read_no_block()
        if rfid_id and rfid_id in rfid_tags:
            return rfid_id, rfid_tags[rfid_id]
        return None, None

    def run(self):
        print("🔑 Access System Running...")
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            now_sec = time.time()
            detected_faces = self.detect_faces(frame)
            rfid_id, rfid_name = self.detect_rfid()

            # Determine if lock can be opened
            if not self.lock_open:
                if detected_faces and now_sec - self.last_close_time >= self.cooldown_seconds:
                    # Open lock for first detected face
                    self.current_user = detected_faces[0]
                    self.open_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.lock_start_time = now_sec
                    self.lock_open = True
                    print(f"🔓 Lock opened by {self.current_user} (Face)")
                    log_open(FACE_CSV, self.current_user)
                    self.other_faces = set(detected_faces[1:])  # log others separately
                elif rfid_id and now_sec - self.last_close_time >= self.cooldown_seconds:
                    self.current_user = rfid_name
                    self.open_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.lock_start_time = now_sec
                    self.lock_open = True
                    print(f"🔓 Lock opened by {self.current_user} (RFID)")
                    log_open(RFID_CSV, self.current_user)
                    self.other_tags = set()
            else:
                # Lock is open: only log other detections, do not affect lock
                for name in detected_faces:
                    if name != self.current_user and name not in self.other_faces:
                        print(f"Other face detected: {name}")
                        log_open(FACE_CSV, name)
                        self.other_faces.add(name)
                if rfid_id and rfid_id not in self.other_tags:
                    print(f"Other RFID detected: {rfid_name}")
                    log_open(RFID_CSV, rfid_name)
                    self.other_tags.add(rfid_id)

                # Auto-close lock after lock_duration
                if now_sec - self.lock_start_time >= self.lock_duration:
                    if (self.current_user in detected_faces) or (self.current_user == rfid_name):
                        GPIO.output(RELAY_GPIO, RELAY_OFF)
                        self.lock_open = False
                        close_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"🔒 Lock closed by {self.current_user}")
                        # Log close
                        if self.current_user in self.face_name_known_list:
                            log_close(FACE_CSV, self.current_user)
                        else:
                            log_close(RFID_CSV, self.current_user)
                        self.current_user = None
                        self.open_time_str = None
                        self.last_close_time = time.time()
                        self.other_faces.clear()
                        self.other_tags.clear()
                    else:
                        # Same user not present → keep waiting
                        self.lock_start_time = now_sec

            cv2.imshow("Access System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.output(RELAY_GPIO, RELAY_OFF)
        GPIO.cleanup()


if __name__ == "__main__":
    system = AccessSystem()
    system.run()
