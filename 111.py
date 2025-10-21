import sys
import os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))
import os
import time
import datetime
import pandas as pd
import sqlite3
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import cv2
import dlib
import numpy as np
from threading import Thread, Lock

# ---------------- GPIO SETUP ----------------
GPIO.setwarnings(False)
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)  # initially closed

# ---------------- FILES & CONFIG ----------------
LOG_FILE = "access_log.csv"
columns = ["Method", "Identifier", "Open Timestamp", "Close Timestamp", "Detected Timestamp"]
csv_mutex = Lock()

MASTER_TAG = "769839607204"
MASTER_NAME = "Universal"

# ---------------- CSV INIT ----------------
def init_csv():
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)

init_csv()

# ---------------- LOGGING ----------------
def log_entry(method, identifier, open_time="", close_time="", detected_time=""):
    with csv_mutex:
        df = pd.read_csv(LOG_FILE)

        if open_time:
            df.loc[len(df)] = [method, identifier, open_time, "", ""]
        elif close_time:
            # update last open entry
            mask = (df["Close Timestamp"] == "") & (df["Open Timestamp"] != "")
            if mask.any():
                idx = df[mask].index[-1]
                df.loc[idx, "Close Timestamp"] = close_time
            else:
                df.loc[len(df)] = [method, identifier, "", close_time, ""]
        elif detected_time:
            df.loc[len(df)] = [method, identifier, "", "", detected_time]
        df.to_csv(LOG_FILE, index=False)

# ---------------- LOCK STATE ----------------
lock_open = False
current_user = None
lock_mutex = Lock()

def open_lock(method, identifier):
    global lock_open, current_user
    with lock_mutex:
        if not lock_open:
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = identifier
            log_entry(method, identifier, open_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"🔓 Lock opened by {identifier}")

def close_lock():
    global lock_open, current_user
    with lock_mutex:
        if lock_open:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            log_entry("RFID", current_user, close_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"🔒 Lock closed by {current_user}")
            lock_open = False
            current_user = None

# ---------------- STARTUP STATE RECOVERY ----------------
def check_last_lock_state():
    global lock_open, current_user
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        mask = (df["Open Timestamp"] != "") & (df["Close Timestamp"] == "")
        if mask.any():
            last = df[mask].iloc[-1]
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = last["Identifier"]
            print(f"🔓 Lock opened on startup by {current_user} (last session not closed)")
        else:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            print("🔒 Lock closed on startup")
    else:
        print("🔒 Lock closed on startup")

check_last_lock_state()

# ---------------- RFID SETUP ----------------
reader = SimpleMFRC522()

def handle_rfid_detection(tag_id):
    global lock_open, current_user

    tag_str = str(tag_id)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ✅ MASTER / UNIVERSAL TAG LOGIC
    if tag_str == MASTER_TAG:
        if lock_open:
            # Close the lock if open
            print(f"📟 Detected {MASTER_NAME} (master) — closing lock...")
            df = pd.read_csv(LOG_FILE)
            if not df.empty:
                mask = (df["Open Timestamp"] != "") & (df["Close Timestamp"] == "")
                if mask.any():
                    idx = df[mask].index[-1]
                    df.loc[idx, "Close Timestamp"] = now
                    df.loc[len(df)] = ["RFID", MASTER_NAME, "", "", now]
                    df.to_csv(LOG_FILE, index=False)
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            print(f"🔒 Lock closed by {MASTER_NAME}")
            lock_open = False
            current_user = None
        else:
            # Open lock if closed
            print(f"📟 Detected {MASTER_NAME} (master) — opening lock...")
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = MASTER_NAME
            df = pd.read_csv(LOG_FILE)
            df.loc[len(df)] = ["RFID", MASTER_NAME, now, "", ""]
            df.to_csv(LOG_FILE, index=False)
            print(f"🔓 Lock opened by {MASTER_NAME}")
        return

    # ✅ NORMAL RFID USERS
    conn = sqlite3.connect("rfid_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?", (tag_str,))
    res = cursor.fetchone()
    conn.close()

    if not res:
        log_entry("RFID", f"Unknown({tag_str})", detected_time=now)
        print(f"⚠️ Unknown RFID {tag_str} detected — logged only")
        return

    name = res[0]
    identifier = f"{name}({tag_str})"

    if lock_open:
        if current_user == identifier:
            # same user tries to close
            close_lock()
        else:
            # log detection
            print(f"📟 Detected {identifier} (lock already opened)")
            log_entry("RFID", identifier, detected_time=now)
    else:
        open_lock("RFID", identifier)

def rfid_thread():
    while True:
        uid, _ = reader.read_no_block()
        if uid:
            handle_rfid_detection(uid)
        time.sleep(0.2)

# ---------------- FACE RECOGNITION ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

known_features, known_names = [], []

def load_face_db():
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            known_names.append(df.iloc[i, 0])
            known_features.append([df.iloc[i, j] for j in range(1, 129)])

load_face_db()

def face_thread():
    global lock_open, current_user
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector(frame, 0)
        for face in faces:
            shape = predictor(frame, face)
            feature = face_model.compute_face_descriptor(frame, shape)
            if known_features:
                distances = [np.linalg.norm(np.array(feature) - np.array(f)) for f in known_features]
                if min(distances) < 0.6:
                    name = known_names[distances.index(min(distances))]
                    identifier = name
                    cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 255, 0), 2)
                    cv2.putText(frame, name, (face.left(), face.top()-10), font, 0.6, (0, 255, 255), 1)
                    if lock_open and current_user == identifier:
                        close_lock()
                    elif not lock_open:
                        open_lock("FACE", identifier)
                    else:
                        log_entry("FACE", identifier, detected_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        print(f"👤 Detected {identifier} (lock already opened)")
                else:
                    log_entry("FACE", "UNKNOWN", detected_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    cv2.putText(frame, "Unknown", (face.left(), face.top()-10), font, 0.6, (0, 0, 255), 1)
        cv2.imshow("Face Access", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)
    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        t1 = Thread(target=rfid_thread)
        t2 = Thread(target=face_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        print("🔒 Lock closed on exit")
