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

# --- CSV logging ---
def log_entry_csv(name, open_time="", close_time=""):
    file_exists = os.path.exists("face_access_log.csv")
    with open("face_access_log.csv", "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Name", "Open Timestamp", "Close Timestamp"])
        writer.writerow([name, open_time, close_time])

class FaceAccessSystem:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.face_features_known_list = []
        self.face_name_known_list = []

        self.lock_open = False
        self.current_user = None
        self.open_time_str = None
        self.last_close_time = 0         # timestamp when lock was last closed
        self.cooldown_seconds = 10       # ignore reopening for 10 seconds after closing

        self.other_faces = {}  # Track other faces detected while lock is open

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
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue

            faces = detector(frame, 0)

            for face in faces:
                shape = predictor(frame, face)
                face_feature = face_reco_model.compute_face_descriptor(frame, shape)
                distances = [self.euclidean_distance(face_feature, f) for f in self.face_features_known_list]

                if distances and min(distances) < 0.6:
                    name = self.face_name_known_list[distances.index(min(distances))]

                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 255), 2)
                    cv2.putText(frame, name, (face.left(), face.top() - 10), self.font, 0.6, (0, 255, 255), 1)

                    now_time = time.time()

                    if not self.lock_open:
                        # Lock closed → open if cooldown passed
                        if now_time - self.last_close_time >= self.cooldown_seconds:
                            GPIO.output(RELAY_GPIO, RELAY_ON)
                            self.lock_open = True
                            self.current_user = name
                            self.open_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"🔓 Lock opened by {name}")
                            log_entry_csv(name, open_time=self.open_time_str)
                    elif self.lock_open and name == self.current_user:
                        # Same person detected → close lock
                        GPIO.output(RELAY_GPIO, RELAY_OFF)
                        self.lock_open = False
                        close_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"🔒 Lock closed by {name}")
                        log_entry_csv(name, open_time=self.open_time_str, close_time=close_time_str)
                        self.current_user = None
                        self.open_time_str = None
                        self.other_faces.clear()
                        self.last_close_time = time.time()
                    else:
                        # Lock open, other faces → log only entry if first time
                        if name not in self.other_faces:
                            entry_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            print(f"Other detected: {name} at {entry_time}")
                            log_entry_csv(name, open_time=entry_time)
                            self.other_faces[name] = entry_time
                else:
                    # Unknown face
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (face.left(), face.top() - 10), self.font, 0.6, (0, 0, 255), 1)

            cv2.imshow("Face Access System", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(0.05)

        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.output(RELAY_GPIO, RELAY_OFF)
        GPIO.cleanup()

if __name__ == "__main__":
    system = FaceAccessSystem()
    system.run()

