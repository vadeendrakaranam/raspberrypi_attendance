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
RELAY_GPIO = 11
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)  # initially closed

# ---------------- CSV LOG ----------------
CSV_FILE = "access_log.csv"
csv_mutex = Lock()
columns = ["Method","Identifier","Open Timestamp","Close Timestamp","Detected Timestamp"]

def init_csv():
    if not os.path.exists(CSV_FILE):
        pd.DataFrame(columns=columns).to_csv(CSV_FILE,index=False)

init_csv()

def log_entry(method, identifier, open_time="", close_time="", detected_time=""):
    with csv_mutex:
        df = pd.read_csv(CSV_FILE)
        # Opened user
        if open_time:
            mask = (df["Method"]==method) & (df["Identifier"]==identifier) & (df["Open Timestamp"]!="") & (df["Close Timestamp"]=="")
            if not mask.any():
                df = pd.concat([df, pd.DataFrame([[method, identifier, open_time, "", ""]], columns=columns)], ignore_index=True)
        # Close the lock
        if close_time:
            mask = (df["Method"]==method) & (df["Identifier"]==identifier) & (df["Open Timestamp"]!="") & (df["Close Timestamp"]=="")
            if mask.any():
                idx = df[mask].index[-1]
                df.at[idx,"Close Timestamp"] = close_time
        # Detected only
        if detected_time:
            df = pd.concat([df, pd.DataFrame([[method, identifier, "", "", detected_time]], columns=columns)], ignore_index=True)
        df.to_csv(CSV_FILE,index=False)

# ---------------- LOCK STATE ----------------
lock_status = "CLOSED"
current_user = None
lock_open_time = 0
ignore_duration = 10  # seconds cooldown
lock_mutex = Lock()
last_close_time = {}

def open_lock(method, identifier):
    global lock_status, current_user, lock_open_time
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with lock_mutex:
        # Cooldown check
        if identifier in last_close_time and time.time() - last_close_time[identifier] < ignore_duration:
            print(f"⏱ Cooldown active for {identifier}, lock not opened")
            log_entry(method, identifier, detected_time=now)
            return
        if lock_status=="CLOSED":
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            lock_status="OPEN"
            current_user = identifier
            lock_open_time = time.time()
            log_entry(method, identifier, open_time=now)
            print(f"🔓 Lock opened by {identifier}")

def close_lock():
    global lock_status, current_user
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with lock_mutex:
        if lock_status=="OPEN" and current_user:
            if time.time() - lock_open_time >= ignore_duration:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
                log_entry(method=current_user.split(":")[0], identifier=current_user, close_time=now)
                print(f"🔒 Lock closed by {current_user}")
                last_close_time[current_user] = time.time()
                current_user = None
                lock_status="CLOSED"

# ---------------- STARTUP RECOVERY ----------------
def check_last_lock_state():
    global lock_status, current_user, lock_open_time
    df = pd.read_csv(CSV_FILE)
    if not df.empty:
        mask = (df["Open Timestamp"]!="") & (df["Close Timestamp"]=="")
        if mask.any():
            last = df[mask].iloc[-1]
            lock_status="OPEN"
            current_user = last["Identifier"]
            lock_open_time = time.time()
            GPIO.output(RELAY_GPIO, GPIO.HIGH)
            print(f"🔓 Lock opened on startup by {current_user} (last session not closed)")
        else:
            GPIO.output(RELAY_GPIO, GPIO.LOW)
            print("🔒 Lock closed on startup")

check_last_lock_state()

# ---------------- RFID ----------------
reader = SimpleMFRC522()
def init_rfid_db():
    conn = sqlite3.connect("rfid_data.db")
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS rfid_users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tag_id TEXT UNIQUE,
                        name TEXT)''')
    conn.commit()
    conn.close()

def get_or_register_rfid(tag_id):
    conn = sqlite3.connect("rfid_data.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?",(str(tag_id),))
    res = cursor.fetchone()
    if res:
        name = res[0]
        conn.close()
        return name
    else:
        name = input(f"🆕 Enter name for new RFID tag {tag_id}: ").strip()
        if name:
            cursor.execute("INSERT INTO rfid_users(tag_id,name) VALUES (?,?)",(str(tag_id),name))
            conn.commit()
        conn.close()
        return name

def rfid_thread():
    global current_user
    while True:
        uid, _ = reader.read_no_block()
        if uid:
            identifier = f"{get_or_register_rfid(uid)}({uid})"
            if current_user==identifier:
                close_lock()
            elif lock_status=="CLOSED":
                open_lock("RFID", identifier)
            else:
                log_entry("RFID", identifier, detected_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                print(f"📟 Detected {identifier} (lock already opened)")
        time.sleep(0.1)

# ---------------- FACE ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

known_features=[]
known_names=[]
def load_face_db():
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv",header=None)
        for i in range(df.shape[0]):
            known_names.append(df.iloc[i,0])
            known_features.append([df.iloc[i,j] for j in range(1,129)])

load_face_db()

def face_thread():
    global current_user
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
            shape = predictor(frame,face)
            feature = face_model.compute_face_descriptor(frame,shape)
            if known_features:
                distances = [np.linalg.norm(np.array(feature)-np.array(f)) for f in known_features]
                if min(distances)<0.6:
                    name = known_names[distances.index(min(distances))]
                    identifier = name
                    cv2.rectangle(frame,(face.left(),face.top()),(face.right(),face.bottom()),(255,255,0),2)
                    cv2.putText(frame,name,(face.left(),face.top()-10),font,0.6,(0,255,255),1)
                    if current_user==identifier:
                        close_lock()
                    elif lock_status=="CLOSED":
                        open_lock("FACE", identifier)
                    else:
                        log_entry("FACE",identifier,detected_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        print(f"👤 Detected {identifier} (lock already opened)")
                else:
                    log_entry("FACE","UNKNOWN",detected_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    cv2.putText(frame,"Unknown",(face.left(),face.top()-10),font,0.6,(0,0,255),1)
        cv2.imshow("Face Access",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
        time.sleep(0.05)
    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__=="__main__":
    init_rfid_db()
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
