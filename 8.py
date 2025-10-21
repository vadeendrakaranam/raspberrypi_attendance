import cv2
import dlib
import numpy as np
import pandas as pd
import datetime
import time
import os
import sqlite3
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
from threading import Thread

# ---------------- GPIO SETUP ----------------
RELAY_PIN = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)

# ---------------- LOG FILES ----------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "access_log.csv")

# ---------------- CREATE NEW CSV IF NEEDED ----------------
def create_new_csv():
    pd.DataFrame(columns=["Method", "Identifier", "Open Timestamp", "Close Timestamp"]).to_csv(LOG_FILE, index=False)

def check_log_age():
    if not os.path.exists(LOG_FILE):
        create_new_csv()
    else:
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(LOG_FILE))
        if (datetime.datetime.now() - last_modified).days >= 50:
            create_new_csv()

check_log_age()

def log_event(method, identifier, opentime, closetime=""):
    df_new = pd.DataFrame([[method, identifier, opentime, closetime]],
                          columns=["Method", "Identifier", "Open Timestamp", "Close Timestamp"])
    df_new.to_csv(LOG_FILE, mode="a", header=False, index=False)
    print(f"🕒 {opentime} | {method} | {identifier} | Open:{opentime} Close:{closetime}")

def update_close_time(method, identifier, close_time):
    if not os.path.exists(LOG_FILE):
        return
    df = pd.read_csv(LOG_FILE)
    # find last row with empty Close Timestamp for same identifier & method
    mask = (df["Method"] == method) & (df["Identifier"] == identifier) & (df["Close Timestamp"].isna())
    if mask.any():
        idx = df[mask].index[-1]
        df.at[idx, "Close Timestamp"] = close_time
        df.to_csv(LOG_FILE, index=False)
        print(f"🔒 Lock closed by {method}:{identifier} at {close_time}")

# ---------------- FACE SETUP ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

def load_face_db():
    face_names, face_features = [], []
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            face_names.append(df.iloc[i][0])
            face_features.append([df.iloc[i][j] for j in range(1, 129)])
    return face_names, face_features

known_names, known_features = load_face_db()

# ---------------- RFID SETUP ----------------
reader = SimpleMFRC522()

def load_rfid_db():
    tags = {}
    conn = sqlite3.connect("rfid_data.db")
    cur = conn.cursor()
    cur.execute('CREATE TABLE IF NOT EXISTS rfid_users (id INTEGER PRIMARY KEY AUTOINCREMENT, tag_id TEXT UNIQUE, name TEXT)')
    for tag_id, name in cur.execute("SELECT tag_id, name FROM rfid_users").fetchall():
        tags[int(tag_id)] = name
    conn.close()
    return tags

rfid_tags = load_rfid_db()

# ---------------- ACCESS CONTROL ----------------
lock_open = False
current_opener = None
last_action_time = 0
ignore_time = 10  # seconds

def open_lock(method, identifier):
    global lock_open, current_opener, last_action_time
    if not lock_open:
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        lock_open = True
        current_opener = f"{method}:{identifier}"
        last_action_time = time.time()
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_event(method, identifier, opentime=now)

def close_lock(method, identifier):
    global lock_open, current_opener
    if lock_open and current_opener == f"{method}:{identifier}" and time.time() - last_action_time >= ignore_time:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        lock_open = False
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        update_close_time(method, identifier, close_time=now)
        current_opener = None

# ---------------- SYNC LOCK ON STARTUP ----------------
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        last_row = df.iloc[-1]
        if pd.isna(last_row["Close Timestamp"]):
            lock_open = True
            current_opener = f"{last_row['Method']}:{last_row['Identifier']}"
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            print(f"🔓 Lock opened on startup (last opened by {current_opener})")
        else:
            GPIO.output(RELAY_PIN, GPIO.LOW)
            print("🔒 Lock is CLOSED on startup")
    else:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        print("🔒 No logs found, lock CLOSED")
else:
    create_new_csv()
    GPIO.output(RELAY_PIN, GPIO.LOW)
    print("🔒 Log file not found, lock CLOSED")

# ---------------- FACE ACCESS THREAD ----------------
def face_access():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    global lock_open, current_opener, last_action_time

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        faces = detector(frame, 0)
        for face in faces:
            shape = predictor(frame, face)
            features = face_rec_model.compute_face_descriptor(frame, shape)
            if known_features:
                distances = [np.linalg.norm(np.array(features) - np.array(f)) for f in known_features]
                min_dist = min(distances)
                if min_dist < 0.6:
                    name = known_names[distances.index(min_dist)]
                    cv2.putText(frame, name, (face.left(), face.top()-10), font, 0.6, (0,255,0), 1)
                    open_lock("FACE", "-")
                    close_lock("FACE", "-")
                else:
                    cv2.putText(frame, "Unknown", (face.left(), face.top()-10), font, 0.6, (0,0,255), 1)
        cv2.imshow("Face Access", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)

    cap.release()
    cv2.destroyAllWindows()

# ---------------- RFID ACCESS THREAD ----------------
def rfid_access():
    global lock_open, current_opener, last_action_time

    while True:
        uid, _ = reader.read_no_block()
        if uid:
            if uid in rfid_tags:
                name = rfid_tags[uid]
                print(f"📟 UID: {uid} | Name: {name}")
                open_lock("RFID", str(uid))
                close_lock("RFID", str(uid))
            else:
                print(f"❌ Unknown RFID UID: {uid}")
        time.sleep(0.1)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        t1 = Thread(target=face_access)
        t2 = Thread(target=rfid_access)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        GPIO.cleanup()
