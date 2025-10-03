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
from threading import Lock

# --- Dlib Models ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# --- Relay GPIO ---
RELAY_GPIO = 11
RELAY_ON = GPIO.HIGH
RELAY_OFF = GPIO.LOW
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, RELAY_OFF)

CSV_FILE = "face_access_log.csv"
LOCK_STATE_FILE = "lock_state.txt"
lock_file_mutex = Lock()  # Thread-safe access

# --- CSV logging ---
def log_open(name, open_time):
    file_exists = os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Open Timestamp", "Close Timestamp"])
        writer.writerow([name, open_time, ""])

def log_close(name, close_time):
    if not os.path.exists(CSV_FILE):
        return
    df = pd.read_csv(CSV_FILE)
    mask = (df["Name"] == name) & (df["Close Timestamp"] == "")
    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "Close Timestamp"] = close_time
        df.to_csv(CSV_FILE, index=False)

# --- Lock state helpers ---
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

class FaceAccessSystem:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.face_features_known_list = []
        self.face_name_known_list = []

        self.other_faces = {}  # Track other faces while lock is open

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
        current_user = None
        lock_open_start_time = 0
        lock_duration = 15  # seconds

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

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

                    now_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Read shared lock state
                    state = get_lock_state()
                    lock_status = state.get("status", "CLOSED")
                    opener = state.get("system", "")

                    if lock_status == "CLOSED":
                        # Open lock for registered face
                        GPIO.output(RELAY_GPIO, RELAY_ON)
                        current_user = name
                        lock_open_start_time = time.time()
                        set_lock_state("OPEN", f"FACE:{name}")
                        print(f"🔓 Lock opened by {name}")
                        log_open(name, now_time_str)

                    elif lock_status == "OPEN" and f"FACE:{name}" != opener:
                        # Lock already open by someone else → log detected face
                        if name not in self.other_faces:
                            print(f"Other detected: {name} at {now_time_str}")
                            log_open(name, now_time_str)
                            self.other_faces[name] = now_time_str

                else:
                    # Unknown face
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (face.left(), face.top() - 10), self.font, 0.6, (0, 0, 255), 1)

            # Automatic close logic
            if current_user:
                elapsed = time.time() - lock_open_start_time
                if elapsed >= lock_duration:
                    if current_user in detected_names:
                        GPIO.output(RELAY_GPIO, RELAY_OFF)
                        close_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"🔒 Lock closed by {current_user}")
                        log_close(current_user, close_time_str)
                        set_lock_state("CLOSED", "NONE")
                        current_user = None
                        self.other_faces.clear()
                    else:
                        # User disappeared → reset start time
                        lock_open_start_time = time.time()

            cv2.imshow("Face Access System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.output(RELAY_GPIO, RELAY_OFF)
        set_lock_state("CLOSED", "NONE")
        GPIO.cleanup()


if __name__ == "__main__":
    system = FaceAccessSystem()
    system.run()
