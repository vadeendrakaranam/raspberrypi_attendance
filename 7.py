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

# ---------------- GPIO ----------------
RELAY_PIN = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)

# ---------------- LOG ----------------
LOG_FILE = "access_log.csv"
log_mutex = Lock()

def create_csv():
    df = pd.DataFrame(columns=["Method","Identifier","Open Timestamp","Close Timestamp"])
    df.to_csv(LOG_FILE,index=False)

def check_csv_age():
    if not os.path.exists(LOG_FILE):
        create_csv()
    else:
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(LOG_FILE))
        if (datetime.datetime.now() - last_modified).days >= 50:
            create_csv()

check_csv_age()

def log_detection(method, identifier, open_time="", close_time=""):
    with log_mutex:
        df = pd.read_csv(LOG_FILE)
        mask = (df["Method"]==method) & (df["Identifier"]==identifier) & (df["Close Timestamp"]=="")
        if mask.any():
            idx = df[mask].index[-1]
            if close_time:
                df.at[idx,"Close Timestamp"] = close_time
        else:
            df = pd.concat([df,pd.DataFrame([[method,identifier,open_time,close_time]],columns=df.columns)],ignore_index=True)
        df.to_csv(LOG_FILE,index=False)

# ---------------- FACE SETUP ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')

def load_face_db():
    names, features = [], []
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            names.append(df.iloc[i][0])
            features.append([df.iloc[i][j] for j in range(1,129)])
    return names, features

known_names, known_features = load_face_db()

# ---------------- RFID SETUP ----------------
reader = SimpleMFRC522()

def init_rfid_db():
    conn = sqlite3.connect('rfid_data.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS rfid_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tag_id TEXT UNIQUE,
            name TEXT
        )''')
    conn.commit()
    conn.close()

def load_rfid_tags():
    tags = {}
    conn = sqlite3.connect('rfid_data.db')
    cursor = conn.cursor()
    for tag_id, name in cursor.execute("SELECT tag_id, name FROM rfid_users").fetchall():
        tags[int(tag_id)] = name
    conn.close()
    return tags

init_rfid_db()
registered_tags = load_rfid_tags()

# ---------------- LOCK CONTROL ----------------
lock_open = False
current_user = None
last_open_time = 0
ignore_time = 10  # seconds cooldown

def open_lock(identifier):
    global lock_open, current_user, last_open_time
    if not lock_open:
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        lock_open = True
        current_user = identifier
        last_open_time = time.time()
        log_detection(identifier.split(":")[0], identifier, open_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"🔓 Lock opened by {identifier}")

def close_lock(identifier):
    global lock_open, current_user
    if lock_open and current_user==identifier and (time.time()-last_open_time)>=ignore_time:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        lock_open = False
        log_detection(identifier.split(":")[0], identifier, close_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"🔒 Lock closed by {identifier}")
        current_user = None

# ---------------- SYNC LOCK STATE ON START ----------------
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        last_row = df.iloc[-1]
        if last_row['Close Timestamp']=="":
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            lock_open = True
            current_user = f"{last_row['Method']}:{last_row['Identifier']}"
            last_open_time = time.time()
            print(f"🔓 Lock opened on startup by {current_user}")
        else:
            GPIO.output(RELAY_PIN, GPIO.LOW)
            print("🔒 Lock closed on startup")
else:
    create_csv()
    GPIO.output(RELAY_PIN, GPIO.LOW)

# ---------------- FACE THREAD ----------------
def face_thread():
    global lock_open, current_user, last_open_time
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret: continue
        faces = detector(frame,0)
        for face in faces:
            shape = predictor(frame, face)
            feature = face_rec_model.compute_face_descriptor(frame,shape)
            identifier = "Unknown"
            if known_features:
                distances = [np.linalg.norm(np.array(feature)-np.array(f)) for f in known_features]
                if min(distances)<0.6:
                    identifier = known_names[distances.index(min(distances))]
            if identifier!="Unknown":
                identifier_full = f"FACE:{identifier}"
                if not lock_open:
                    open_lock(identifier_full)
                else:
                    close_lock(identifier_full)
            else:
                log_detection("FACE","Unknown")  # only log
        cv2.imshow("Face Access", frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break
        time.sleep(0.05)
    cap.release()
    cv2.destroyAllWindows()

# ---------------- RFID THREAD ----------------
def rfid_thread():
    global lock_open, current_user, last_open_time
    while True:
        uid, _ = reader.read_no_block()
        if uid:
            if uid in registered_tags:
                identifier = f"RFID:{registered_tags[uid]}({uid})"
                if not lock_open:
                    open_lock(identifier)
                else:
                    close_lock(identifier)
            else:
                log_detection("RFID","Unknown")
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
