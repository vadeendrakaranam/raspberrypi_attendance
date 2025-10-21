import os
import cv2
import dlib
import numpy as np
import pandas as pd
import datetime
import time
import sqlite3
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
from threading import Thread, Lock

# ---------------- GPIO SETUP ----------------
RELAY_PIN = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)  # Lock closed initially

# ---------------- CSV LOG ----------------
LOG_FILE = "access_log.csv"
log_mutex = Lock()

# Ensure CSV exists and clean
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=["Method","Identifier","Open Timestamp","Close Timestamp"]).to_csv(LOG_FILE,index=False)

# ---------------- LOG FUNCTIONS ----------------
def log_detection(method, identifier, open_time="", close_time=""):
    """Log access detection in a single CSV row per open-close cycle."""
    with log_mutex:
        # read existing CSV with only expected columns
        df = pd.read_csv(LOG_FILE, usecols=["Method","Identifier","Open Timestamp","Close Timestamp"])
        
        # check if open row exists
        mask = (df["Method"]==method) & (df["Identifier"]==identifier) & (df["Close Timestamp"]=="")
        if mask.any():
            idx = df[mask].index[-1]
            if close_time:
                df.at[idx,"Close Timestamp"] = close_time
        else:
            # append new row
            new_row = pd.DataFrame([[method, identifier, open_time, close_time]],
                                   columns=["Method","Identifier","Open Timestamp","Close Timestamp"])
            df = pd.concat([df,new_row], ignore_index=True)
        
        df.to_csv(LOG_FILE, index=False)

def get_last_open_status():
    """Check last open row to restore lock after reboot."""
    df = pd.read_csv(LOG_FILE, usecols=["Method","Identifier","Open Timestamp","Close Timestamp"])
    if df.empty:
        return None,None
    last_open = df[df["Close Timestamp"]==""]
    if last_open.empty:
        return None,None
    last_row = last_open.iloc[-1]
    return last_row["Method"], last_row["Identifier"]

# ---------------- LOCK CONTROL ----------------
lock_open = False
current_user = None
lock_time = 0
IGNORE_DURATION = 10  # seconds

def open_lock(method, identifier):
    global lock_open, current_user, lock_time
    if not lock_open:
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        lock_open = True
        current_user = identifier
        lock_time = time.time()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_detection(method, identifier, open_time=now)
        print(f"🔓 Lock OPENED by {identifier}")

def close_lock(method, identifier):
    global lock_open, current_user
    if lock_open and current_user==identifier and time.time()-lock_time>=IGNORE_DURATION:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_detection(method, identifier, close_time=now)
        print(f"🔒 Lock CLOSED by {identifier}")
        lock_open = False
        current_user = None

# ---------------- FACE SYSTEM ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

class FaceAccessSystem(Thread):
    def __init__(self):
        super().__init__()
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.known_names = []
        self.known_features = []
        self.last_closed_time = {}
        self.load_face_database()

    def load_face_database(self):
        if not os.path.exists("data/features_all.csv"):
            print("❌ Face database not found. Exiting.")
            exit()
        df = pd.read_csv("data/features_all.csv",header=None)
        for i in range(df.shape[0]):
            self.known_names.append(df.iloc[i][0])
            self.known_features.append([df.iloc[i][j] for j in range(1,129)])

    def euclidean_distance(self,f1,f2):
        return np.linalg.norm(np.array(f1)-np.array(f2))

    def run(self):
        global lock_open
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            faces = detector(frame,0)
            for face in faces:
                shape = predictor(frame, face)
                feature = face_reco_model.compute_face_descriptor(frame, shape)
                distances = [self.euclidean_distance(feature,f) for f in self.known_features]
                if distances and min(distances)<0.6:
                    name = self.known_names[distances.index(min(distances))]
                    cv2.putText(frame,name,(face.left(),face.top()-10),self.font,0.6,(0,255,0),1)
                    # open or close
                    state = get_last_open_status()
                    if not lock_open:
                        if name not in self.last_closed_time or time.time()-self.last_closed_time[name]>IGNORE_DURATION:
                            open_lock("FACE",name)
                    elif current_user==name:
                        close_lock("FACE",name)
                    else:
                        # other face detected while lock open
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_detection("FACE",name,open_time="", close_time=now)
                        print(f"❌ Other face detected: {name}")
                else:
                    # unknown face
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_detection("FACE","UNKNOWN",open_time="",close_time=now)
                    print(f"❌ Unknown face detected")
            cv2.imshow("Face Access",frame)
            if cv2.waitKey(1)&0xFF==ord('q'):
                break
            time.sleep(0.05)
        self.cap.release()
        cv2.destroyAllWindows()

# ---------------- RFID SYSTEM ----------------
class RFIDAccessSystem(Thread):
    def __init__(self):
        super().__init__()
        self.reader = SimpleMFRC522()
        self.rfid_tags = self.load_rfid_db()
        self.last_closed_time = {}

    def load_rfid_db(self):
        tags = {}
        conn = sqlite3.connect("rfid_data.db")
        cursor = conn.cursor()
        cursor.execute("""CREATE TABLE IF NOT EXISTS rfid_users (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            tag_id TEXT UNIQUE,
                            name TEXT)""")
        for tag_id,name in cursor.execute("SELECT tag_id,name FROM rfid_users"):
            tags[int(tag_id)] = name
        conn.close()
        return tags

    def run(self):
        global lock_open
        while True:
            uid, _ = self.reader.read_no_block()
            if uid:
                if uid in self.rfid_tags:
                    name = self.rfid_tags[uid]
                    if not lock_open:
                        if uid not in self.last_closed_time or time.time()-self.last_closed_time[uid]>IGNORE_DURATION:
                            open_lock("RFID",f"{name}({uid})")
                    elif current_user==f"{name}({uid})":
                        close_lock("RFID",f"{name}({uid})")
                    else:
                        # other tag detected while lock open
                        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        log_detection("RFID",f"{name}({uid})",open_time="",close_time=now)
                        print(f"❌ Other RFID detected: {name}({uid})")
                else:
                    # unknown tag
                    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_detection("RFID","UNKNOWN",open_time="",close_time=now)
                    print(f"❌ Unknown RFID detected: {uid}")
            time.sleep(0.1)

# ---------------- MAIN ----------------
if __name__=="__main__":
    # restore lock state after reboot
    last_method, last_id = get_last_open_status()
    if last_method and last_id:
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        lock_open = True
        current_user = last_id
        lock_time = time.time()
        print(f"🔓 Lock restored OPEN for {current_user} after reboot")
    else:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        print("🔒 Lock closed on startup")

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
        GPIO.output(RELAY_PIN, GPIO.LOW)
