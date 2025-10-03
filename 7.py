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

# --- GPIO setup ---
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)  # lock initially closed

# --- Dlib Models ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# --- RFID setup ---
reader = SimpleMFRC522()

# --- CSV files ---
FACE_CSV = "face_access_log.csv"
RFID_CSV = "rfid_access_log.csv"

# --- Logging functions ---
def log_open(csv_file, name, open_time):
    file_exists = os.path.exists(csv_file)
    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Open Timestamp", "Close Timestamp"])
        writer.writerow([name, open_time, ""])

def log_close(csv_file, name, close_time):
    if not os.path.exists(csv_file):
        return
    df = pd.read_csv(csv_file)
    mask = (df["Name"] == name) & (df["Close Timestamp"] == "")
    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "Close Timestamp"] = close_time
        df.to_csv(csv_file, index=False)

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

# --- Face Access System ---
class AccessSystem:
    def __init__(self):
        # Face DB
        self.face_name_known_list = []
        self.face_features_known_list = []
        self.load_face_database()

        # Lock state
        self.lock_open = False
        self.current_user = None
        self.open_time_str = None
        self.lock_start_time = 0
        self.lock_duration = 15  # seconds
        self.cooldown_after_close = 10
        self.last_close_time = 0

        self.other_faces = set()
        self.other_tags = set()

        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    def load_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                features = [csv_rd.iloc[i][j] for j in range(1, 129)]
                self.face_features_known_list.append(features)

    def euclidean_distance(self, f1, f2):
        return np.linalg.norm(np.array(f1) - np.array(f2))

    def run(self):
        print("🔑 Access System Running...")
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    continue

                detected_names = []

                # --- Face detection ---
                faces = detector(frame, 0)
                for face in faces:
                    shape = predictor(frame, face)
                    face_feature = face_reco_model.compute_face_descriptor(frame, shape)
                    distances = [self.euclidean_distance(face_feature, f) for f in self.face_features_known_list]

                    if distances and min(distances) < 0.6:
                        name = self.face_name_known_list[distances.index(min(distances))]
                        detected_names.append(name)
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
                        cv2.putText(frame, name, (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255),1)

                        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        now_sec = time.time()

                        # Open lock if closed and cooldown passed
                        if not self.lock_open and now_sec - self.last_close_time >= self.cooldown_after_close:
                            GPIO.output(RELAY_GPIO, GPIO.HIGH)
                            self.lock_open = True
                            self.current_user = name
                            self.open_time_str = now_str
                            self.lock_start_time = now_sec
                            print(f"🔓 Lock opened by {name} (Face)")
                            log_open(FACE_CSV, name, self.open_time_str)
                            self.other_faces = set()

                        # Log other faces
                        elif self.lock_open and name != self.current_user:
                            if name not in self.other_faces:
                                print(f"Other face detected: {name}")
                                log_open(FACE_CSV, name, now_str)
                                self.other_faces.add(name)

                    else:
                        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,0,255),2)
                        cv2.putText(frame, "Unknown", (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255),1)

                # --- RFID detection ---
                try:
                    rfid_id, text = reader.read_no_block()
                except:
                    rfid_id = None

                if rfid_id:
                    if rfid_id in rfid_tags:
                        rfid_name = rfid_tags[rfid_id]
                        detected_names.append(rfid_name)
                        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        now_sec = time.time()

                        if not self.lock_open and now_sec - self.last_close_time >= self.cooldown_after_close:
                            GPIO.output(RELAY_GPIO, GPIO.HIGH)
                            self.lock_open = True
                            self.current_user = rfid_name
                            self.open_time_str = now_str
                            self.lock_start_time = now_sec
                            print(f"🔓 Lock opened by {rfid_name} (RFID)")
                            log_open(RFID_CSV, rfid_name, self.open_time_str)
                            self.other_tags = set()

                        elif self.lock_open and rfid_name != self.current_user:
                            if rfid_id not in self.other_tags:
                                print(f"Other RFID detected: {rfid_name}")
                                log_open(RFID_CSV, rfid_name, now_str)
                                self.other_tags.add(rfid_id)

                # --- Automatic close ---
                if self.lock_open:
                    if time.time() - self.lock_start_time >= self.lock_duration:
                        # Close lock only if current_user is still present
                        if self.current_user in detected_names:
                            GPIO.output(RELAY_GPIO, GPIO.LOW)
                            self.lock_open = False
                            close_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"🔒 Lock closed by {self.current_user}")
                            # Log closing
                            log_close(FACE_CSV, self.current_user)
                            log_close(RFID_CSV, self.current_user)
                            self.current_user = None
                            self.open_time_str = None
                            self.other_faces.clear()
                            self.other_tags.clear()
                            self.last_close_time = time.time()
                        else:
                            # User disappeared → continue waiting
                            self.lock_start_time = time.time()

                cv2.imshow("Access System", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            GPIO.cleanup()


if __name__ == "__main__":
    system = AccessSystem()
    system.run()
