import sys
import os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))
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
import json

# ---------------- GPIO SETUP ----------------
GPIO.setwarnings(False)
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)  # initially closed

# ---------------- FILES & CONFIG ----------------
LOG_FILE = "access_log.csv"
COOLDOWN_FILE = "cooldown.json"
columns = ["Method", "Identifier", "Open Timestamp", "Close Timestamp", "Detected Timestamp"]
csv_mutex = Lock()

MASTER_TAG = "769839607204"
MASTER_NAME = "Universal"

# ---------------- COOL DOWN CONFIG ----------------
COOLDOWN_TIME = 10  # seconds per user
user_last_action = {}  # tracks last action time per user
cooldown_mutex = Lock()

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

def save_cooldown():
    with cooldown_mutex:
        with open(COOLDOWN_FILE, "w") as f:
            json.dump(user_last_action, f)

def load_cooldown():
    global user_last_action
    if os.path.exists(COOLDOWN_FILE):
        with open(COOLDOWN_FILE, "r") as f:
            try:
                user_last_action = json.load(f)
            except:
                user_last_action = {}

def open_lock(method, identifier):
    global lock_open, current_user, user_last_action
    with lock_mutex:
        last_time = user_last_action.get(identifier, 0)
        if time.time() - last_time < COOLDOWN_TIME:
            print(f"⏳ Cooldown active for {identifier} — ignoring open request")
            return

        if not lock_open:
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = identifier
            with cooldown_mutex:
                user_last_action[identifier] = time.time()
            log_entry(method, identifier, open_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"🔓 Lock opened by {identifier}")

def close_lock():
    global lock_open, current_user, user_last_action
    with lock_mutex:
        if current_user is None:
            return

        last_time = user_last_action.get(current_user, 0)
        if time.time() - last_time < COOLDOWN_TIME:
            print(f"⏳ Cooldown active for {current_user} — ignoring close request")
            return

        if lock_open:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            log_entry("RFID", current_user, close_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(f"🔒 Lock closed by {current_user}")
            with cooldown_mutex:
                user_last_action[current_user] = time.time()
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
            with cooldown_mutex:
                user_last_action[current_user] = time.time()
            print(f"🔓 Lock opened on startup by {current_user} (last session not closed)")
        else:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            print("🔒 Lock closed on startup")
    else:
        print("🔒 Lock closed on startup")

check_last_lock_state()
load_cooldown()

# ---------------- RFID SETUP ----------------
reader = SimpleMFRC522()

def handle_rfid_detection(tag_id):
    global lock_open, current_user, user_last_action

    tag_str = str(tag_id)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # MASTER TAG LOGIC
    if tag_str == MASTER_TAG:
        last_time = user_last_action.get(MASTER_NAME, 0)
        if time.time() - last_time < COOLDOWN_TIME:
            print(f"⏳ Cooldown active for {MASTER_NAME} — ignoring request")
            return

        if lock_open:
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
            print(f"📟 Detected {MASTER_NAME} (master) — opening lock...")
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_open = True
            current_user = MASTER_NAME
            df = pd.read_csv(LOG_FILE)
            df.loc[len(df)] = ["RFID", MASTER_NAME, now, "", ""]
            df.to_csv(LOG_FILE, index=False)
            print(f"🔓 Lock opened by {MASTER_NAME}")
        with cooldown_mutex:
            user_last_action[MASTER_NAME] = time.time()
        return

    # NORMAL RFID USERS
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
            close_lock()
        else:
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
    global lock_open, current_user, user_last_action
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
                    last_time = user_last_action.get(identifier, 0)
                    if lock_open and current_user == identifier:
                        if time.time() - last_time >= COOLDOWN_TIME:
                            close_lock()
                        else:
                            print(f"⏳ Cooldown active for {identifier} — ignoring close")
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

# ---------------- PERSIST COOLDOWN PERIODICALLY ----------------
def cooldown_saver_thread():
    while True:
        save_cooldown()
        time.sleep(5)  # save every 5 seconds

# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        t1 = Thread(target=rfid_thread)
        t2 = Thread(target=face_thread)
        t3 = Thread(target=cooldown_saver_thread, daemon=True)
        t1.start()
        t2.start()
        t3.start()
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        save_cooldown()
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        print("🔒 Lock closed on exit")
