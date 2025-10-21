import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import dlib
import numpy as np
import cv2
import pandas as pd
import datetime
import time
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
from threading import Lock, Thread

# ---------------- GPIO SETUP ----------------
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)

# ---------------- LOG FILE ----------------
LOG_FILE = "access_log.csv"
log_mutex = Lock()

def init_log():
    """Create CSV if not exists or clear if older than 50 days"""
    if not os.path.exists(LOG_FILE):
        pd.DataFrame(columns=["Method","Identifier","Detected Timestamp","Open Timestamp","Close Timestamp"]).to_csv(LOG_FILE, index=False)
    else:
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(LOG_FILE))
        if (datetime.datetime.now() - last_modified).days >= 50:
            pd.DataFrame(columns=["Method","Identifier","Detected Timestamp","Open Timestamp","Close Timestamp"]).to_csv(LOG_FILE, index=False)

init_log()

def log_detect(method, identifier):
    with log_mutex:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_new = pd.DataFrame([[method, identifier, now, "", ""]],
                              columns=["Method","Identifier","Detected Timestamp","Open Timestamp","Close Timestamp"])
        df_new.to_csv(LOG_FILE, mode="a", header=False, index=False)
        print(f"ℹ️ {now} | {method} | {identifier} | DETECTED")
        return now

def log_open_close(method, identifier, open_time="", close_time=""):
    with log_mutex:
        df = pd.read_csv(LOG_FILE)
        # Update last row without close time if exists
        mask = (df["Method"]==method) & (df["Identifier"]==identifier) & (df["Open Timestamp"]!="" ) & (df["Close Timestamp"]=="")
        if open_time and mask.any():
            idx = df[mask].index[-1]
            df.at[idx,"Open Timestamp"] = open_time
        elif open_time:
            df_new = pd.DataFrame([[method, identifier, "", open_time, ""]],
                                  columns=["Method","Identifier","Detected Timestamp","Open Timestamp","Close Timestamp"])
            df_new.to_csv(LOG_FILE, mode="a", header=False, index=False)
        if close_time and mask.any():
            idx = df[mask].index[-1]
            df.at[idx,"Close Timestamp"] = close_time
        df.to_csv(LOG_FILE, index=False)

# ---------------- DLIB SETUP ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# Load known face database
def load_face_db():
    names, features = [], []
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            names.append(df.iloc[i][0])
            features.append([df.iloc[i][j] for j in range(1,129)])
    return names, features

known_names, known_features = load_face_db()

def euclidean_distance(f1,f2):
    return np.linalg.norm(np.array(f1)-np.array(f2))

# ---------------- LOCK STATE ----------------
lock_open = False
current_opener = None
last_action_time = 0
ignore_time = 10  # seconds

def open_lock(method, identifier):
    global lock_open, current_opener, last_action_time
    if not lock_open:
        GPIO.output(RELAY_GPIO, GPIO.HIGH)
        lock_open = True
        current_opener = f"{method}:{identifier}"
        last_action_time = time.time()
        log_open_close(method, identifier, open_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"🔓 Lock opened by {method}:{identifier}")

def close_lock():
    global lock_open, current_opener
    if lock_open and time.time()-last_action_time>=ignore_time:
        method, identifier = current_opener.split(":",1)
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        log_open_close(method, identifier, close_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"🔒 Lock closed by {method}:{identifier}")
        lock_open = False
        current_opener = None

# ---------------- RESTORE LOCK ON START ----------------
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        last_open = df[df["Close Timestamp"]==""]
        if not last_open.empty:
            last_row = last_open.iloc[-1]
            lock_open = True
            current_opener = f"{last_row['Method']}:{last_row['Identifier']}"
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            print(f"🔓 Lock restored on startup by {current_opener}")
        else:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            print("🔒 Lock CLOSED on startup")
else:
    GPIO.output(RELAY_GPIO, GPIO.LOW)
    print("🔒 Lock CLOSED on startup")

# ---------------- FACE THREAD ----------------
def face_thread():
    global lock_open, current_opener, last_action_time
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector(frame,0)
        for face in faces:
            shape = predictor(frame, face)
            feature = face_reco_model.compute_face_descriptor(frame, shape)
            name = "Unknown"
            if known_features:
                distances = [euclidean_distance(feature,f) for f in known_features]
                if min(distances)<0.6:
                    name = known_names[distances.index(min(distances))]
            log_detect("FACE", name)
            # open only if lock is closed
            if not lock_open:
                if name!="Unknown":
                    open_lock("FACE", name)
            elif current_opener=="FACE:"+name:
                close_lock()
        time.sleep(0.05)

# ---------------- RFID THREAD ----------------
reader = SimpleMFRC522()
rfid_tags = {}  # tag_id:int -> Name:str

def init_rfid_db():
    conn = sqlite3.connect('rfid_data.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS rfid_users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tag_id TEXT UNIQUE,
                        name TEXT)''')
    conn.commit()
    # load into dictionary
    cursor.execute("SELECT tag_id,name FROM rfid_users")
    for tag_id, name in cursor.fetchall():
        rfid_tags[int(tag_id)] = name
    conn.close()

init_rfid_db()

def register_rfid(tag_id):
    conn = sqlite3.connect('rfid_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?",(str(tag_id),))
    res = cursor.fetchone()
    if res:
        name = res[0]
    else:
        name = input(f"Enter name for new RFID tag {tag_id}: ").strip()
        cursor.execute("INSERT INTO rfid_users(tag_id,name) VALUES (?,?)",(str(tag_id),name))
        conn.commit()
        rfid_tags[int(tag_id)] = name
    conn.close()
    return name

def rfid_thread():
    global lock_open, current_opener, last_action_time
    while True:
        uid, _ = reader.read_no_block()
        if uid:
            if uid not in rfid_tags:
                name = register_rfid(uid)
            else:
                name = rfid_tags[uid]
            log_detect("RFID", f"{name}({uid})")
            if not lock_open:
                open_lock("RFID", f"{name}({uid})")
            elif current_opener==f"RFID:{name}({uid})":
                close_lock()
        time.sleep(0.1)

# ---------------- MAIN ----------------
if __name__=="__main__":
    try:
        t_face = Thread(target=face_thread)
        t_rfid = Thread(target=rfid_thread)
        t_face.start()
        t_rfid.start()
        t_face.join()
        t_rfid.join()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        GPIO.cleanup()
