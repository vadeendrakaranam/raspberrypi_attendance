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

# --- Relay GPIO ---
RELAY_GPIO = 11
RELAY_ON = GPIO.HIGH
RELAY_OFF = GPIO.LOW
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, RELAY_OFF)

# --- CSV log ---
ACCESS_CSV_FILE = "access_log.csv"

def log_open(identifier, source, open_time):
    file_exists = os.path.exists(ACCESS_CSV_FILE)
    with open(ACCESS_CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["ID/Name", "Source", "Open Timestamp", "Close Timestamp"])
        writer.writerow([identifier, source, open_time, ""])

def log_close(identifier, source, close_time):
    if not os.path.exists(ACCESS_CSV_FILE):
        return
    df = pd.read_csv(ACCESS_CSV_FILE)
    mask = (df["ID/Name"] == identifier) & (df["Source"] == source) & (df["Close Timestamp"] == "")
    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "Close Timestamp"] = close_time
        df.to_csv(ACCESS_CSV_FILE, index=False)

# --- RFID setup ---
reader = SimpleMFRC522()

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

# --- Face Recognition setup ---
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
        self.current_source = None
        self.open_time_str = None
        self.lock_open_start_time = 0
        self.lock_duration = 15       # Lock stays open
        self.cooldown_seconds = 15    # Cooldown after closing
        self.last_close_time = 0

        self.other_faces = {}
        self.other_tags_detected = set()

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

    def run(self):
        print("🔑 Integrated Access System Running...")
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                detected_names = []

                # --- Face Recognition ---
                faces = detector(frame, 0)
                for face in faces:
                    shape = predictor(frame, face)
                    face_feature = face_reco_model.compute_face_descriptor(frame, shape)
                    distances = [self.euclidean_distance(face_feature, f) for f in self.face_features_known_list]

                    if distances and min(distances) < 0.6:
                        name = self.face_name_known_list[distances.index(min(distances))]
                        detected_names.append(name)
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
                        cv2.putText(frame, name, (face.left(), face.top()-10), self.font, 0.6, (0,255,255), 1)

                        now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        now_time_sec = time.time()

                        # Lock opening for face
                        if not self.lock_open and now_time_sec - self.last_close_time >= self.cooldown_seconds:
                            GPIO.output(RELAY_GPIO, RELAY_ON)
                            self.lock_open = True
                            self.current_user = name
                            self.current_source = "Face"
                            self.open_time_str = now_time_str
                            self.lock_open_start_time = now_time_sec
                            print(f"🔓 Lock opened by FACE: {name}")
                            log_open(name, "Face", now_time_str)

                        # Log other faces
                        elif self.lock_open and name != self.current_user:
                            if name not in self.other_faces:
                                log_open(name, "Face", now_time_str)
                                self.other_faces[name] = now_time_str

                    else:
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,0,255), 2)
                        cv2.putText(frame, "Unknown", (face.left(), face.top()-10), self.font, 0.6, (0,0,255), 1)

                # --- RFID Detection ---
                rfid_id, text = reader.read_no_block()
                if rfid_id:
                    if rfid_id in rfid_tags:
                        tag_name = rfid_tags[rfid_id]
                        now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        now_time_sec = time.time()

                        if not self.lock_open and now_time_sec - self.last_close_time >= self.cooldown_seconds:
                            GPIO.output(RELAY_GPIO, RELAY_ON)
                            self.lock_open = True
                            self.current_user = tag_name
                            self.current_source = "RFID"
                            self.open_time_str = now_time_str
                            self.lock_open_start_time = now_time_sec
                            print(f"🔓 Lock opened by RFID: {tag_name}")
                            log_open(tag_name, "RFID", now_time_str)

                        elif self.lock_open and self.current_user != tag_name:
                            if rfid_id not in self.other_tags_detected:
                                log_open(tag_name, "RFID", now_time_str)
                                self.other_tags_detected.add(rfid_id)

                    else:
                        print(f"❌ Unknown RFID tag: {rfid_id}")

                # --- Automatic Close ---
                if self.lock_open:
                    elapsed = time.time() - self.lock_open_start_time
                    if elapsed >= self.lock_duration:
                        # Close lock
                        GPIO.output(RELAY_GPIO, RELAY_OFF)
                        self.lock_open = False
                        close_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"🔒 Lock closed: {self.current_user} ({self.current_source})")
                        log_close(self.current_user, self.current_source, close_time_str)
                        self.current_user = None
                        self.current_source = None
                        self.open_time_str = None
                        self.other_faces.clear()
                        self.other_tags_detected.clear()
                        self.last_close_time = time.time()

                cv2.imshow("Access System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.05)

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            GPIO.output(RELAY_GPIO, RELAY_OFF)
            GPIO.cleanup()

if __name__ == "__main__":
    system = AccessSystem()
    system.run()
