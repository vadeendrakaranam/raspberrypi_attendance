import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

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
from threading import Thread, Lock

# ---------------- GPIO SETUP ----------------
RELAY_PIN = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)

# ---------------- LOG SETUP ----------------
LOG_FILE = "access_log.csv"
log_mutex = Lock()
os.makedirs(os.path.dirname(LOG_FILE) or ".", exist_ok=True)

if not os.path.exists(LOG_FILE) or os.stat(LOG_FILE).st_size == 0:
    # Create CSV with headers if empty
    pd.DataFrame(columns=["Method","Identifier","Open Timestamp","Close Timestamp"]).to_csv(LOG_FILE, index=False)

def log_open(method, identifier):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_mutex:
        df = pd.read_csv(LOG_FILE)
        df.loc[len(df)] = [method, identifier, now, ""]
        df.to_csv(LOG_FILE, index=False)
    print(f"🔓 {now} | {method} | {identifier} | OPEN")
    return now

def log_detected(method, identifier):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_mutex:
        df = pd.read_csv(LOG_FILE)
        df.loc[len(df)] = [method, identifier, "", now]  # only detected
        df.to_csv(LOG_FILE, index=False)
    print(f"👁️ {now} | {method} | {identifier} | DETECTED")
    return now

def log_close(method, identifier):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with log_mutex:
        df = pd.read_csv(LOG_FILE)
        mask = (df["Method"]==method) & (df["Identifier"]==identifier) & (df["Close Timestamp"]=="")
        if mask.any():
            idx = df[mask].index[-1]
            df.at[idx,"Close Timestamp"] = now
            df.to_csv(LOG_FILE, index=False)
            print(f"🔒 {now} | {method} | {identifier} | CLOSE")
            return True
    return False

# ---------------- LOCK STATE ----------------
lock_open = False
current_opener = None
ignore_time = 10
last_action_time = 0

def open_lock(method, identifier):
    global lock_open, current_opener, last_action_time
    if not lock_open:
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        lock_open = True
        current_opener = f"{method}:{identifier}"
        last_action_time = time.time()
        log_open(method, identifier)

def close_lock(method, identifier):
    global lock_open, current_opener, last_action_time
    if lock_open and current_opener == f"{method}:{identifier}" and time.time()-last_action_time >= ignore_time:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        lock_open = False
        log_close(method, identifier)
        current_opener = None

# ---------------- SYNC LOCK ON STARTUP ----------------
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        last_row = df.iloc[-1]
        if last_row['Open Timestamp'] != "" and last_row['Close Timestamp']=="":
            # last open not closed → reopen lock
            lock_open = True
            current_opener = f"{last_row['Method']}:{last_row['Identifier']}"
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            print(f"🔓 Lock reopened on startup by {current_opener}")
        else:
            print("🔒 Lock CLOSED on startup")

# ---------------- FACE SETUP ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')

# Load face DB
def load_face_db():
    names, features = [], []
    if os.path.exists("data/features_all.csv") and os.stat("data/features_all.csv").st_size > 0:
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            names.append(df.iloc[i][0])
            features.append([df.iloc[i][j] for j in range(1,129)])
    return names, features

known_names, known_features = load_face_db()

# ---------------- RFID SETUP ----------------
reader = SimpleMFRC522()
db_file = 'rfid_data.db'
conn = sqlite3.connect(db_file)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS rfid_users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tag_id TEXT UNIQUE,
    name TEXT
)''')
conn.commit()

def get_or_register_rfid(tag_id):
    cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?", (str(tag_id),))
    res = cursor.fetchone()
    if res:
        return res[0]
    else:
        name = input(f"🆕 Enter name for new RFID tag {tag_id}: ").strip()
        if name:
            cursor.execute("INSERT INTO rfid_users(tag_id,name) VALUES (?,?)", (str(tag_id),name))
            conn.commit()
            return name
        else:
            return None

# ---------------- FACE THREAD ----------------
def face_thread():
    global lock_open, current_opener, last_action_time
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector(frame,0)
        for face in faces:
            shape = predictor(frame, face)
            feature = face_rec_model.compute_face_descriptor(frame, shape)
            if known_features:
                distances = [np.linalg.norm(np.array(feature)-np.array(f)) for f in known_features]
                if min(distances)<0.6:
                    name = known_names[distances.index(min(distances))]
                    cv2.putText(frame,name,(face.left(),face.top()-10),font,0.6,(0,255,0),1)
                    if not lock_open:
                        open_lock("FACE", name)
                    elif current_opener == f"FACE:{name}" and time.time()-last_action_time>=ignore_time:
                        close_lock("FACE", name)
                else:
                    cv2.putText(frame,"Unknown",(face.left(),face.top()-10),font,0.6,(0,0,255),1)
                    log_detected("FACE","Unknown")
            else:
                log_detected("FACE","Unknown")

        cv2.imshow("Face Access", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.05)
    cap.release()
    cv2.destroyAllWindows()

# ---------------- RFID THREAD ----------------
def rfid_thread():
    global lock_open, current_opener, last_action_time
    while True:
        uid, _ = reader.read_no_block()
        if uid:
            name = get_or_register_rfid(uid)
            if not name:
                name = "Unknown"
            identifier = f"{name}({uid})"
            if not lock_open:
                open_lock("RFID", identifier)
            elif current_opener == f"RFID:{identifier}" and time.time()-last_action_time>=ignore_time:
                close_lock("RFID", identifier)
            else:
                log_detected("RFID", identifier)
        time.sleep(0.1)

# ---------------- MAIN ----------------
if __name__=="__main__":
    try:
        t1 = Thread(target=face_thread)
        t2 = Thread(target=rfid_thread)
        t1.start()
        t2.start()
        t1.join()
        t2.join()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        GPIO.cleanup()
        conn.close()
