import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import cv2, dlib, numpy as np, pandas as pd, datetime, time
import sqlite3, RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
from threading import Thread, Lock

# ---------------- GPIO SETUP ----------------
RELAY_PIN = 11
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)

# ---------------- LOG FILE ----------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, "access_log.csv")

log_mutex = Lock()
lock_mutex = Lock()
lock_open = False
current_opener = None
ignore_time = 10  # seconds
last_action_time = 0

def create_new_csv():
    pd.DataFrame(columns=["Method","Identifier","Open Timestamp","Close Timestamp"]).to_csv(LOG_FILE, index=False)

def check_log_age():
    if not os.path.exists(LOG_FILE):
        create_new_csv()
    else:
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(LOG_FILE))
        if (datetime.datetime.now() - last_modified).days >= 50:
            create_new_csv()

check_log_age()

def log_open(method, identifier):
    with log_mutex:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        pd.DataFrame([[method, identifier, now, ""]],
                     columns=["Method","Identifier","Open Timestamp","Close Timestamp"]).to_csv(LOG_FILE, mode="a", header=False, index=False)
        print(f"🔓 {now} | {method} | {identifier} | OPEN")
        return now

def log_close(method, identifier):
    with log_mutex:
        if not os.path.exists(LOG_FILE):
            return
        df = pd.read_csv(LOG_FILE)
        mask = (df["Method"]==method) & (df["Identifier"]==identifier) & (df["Close Timestamp"]=="")
        if mask.any():
            idx = df[mask].index[-1]
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df.at[idx,"Close Timestamp"]=now
            df.to_csv(LOG_FILE,index=False)
            print(f"🔒 {now} | {method} | {identifier} | CLOSE")

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
    cursor.execute('''CREATE TABLE IF NOT EXISTS rfid_users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        tag_id TEXT UNIQUE,
                        name TEXT)''')
    conn.commit()
    conn.close()

def load_rfid_tags():
    tags = {}
    conn = sqlite3.connect('rfid_data.db')
    cursor = conn.cursor()
    for tag_id, name in cursor.execute("SELECT tag_id,name FROM rfid_users").fetchall():
        tags[int(tag_id)] = name
    conn.close()
    return tags

def register_rfid_tag(tag_id):
    conn = sqlite3.connect('rfid_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM rfid_users WHERE tag_id=?", (str(tag_id),))
    res = cursor.fetchone()
    if res:
        conn.close()
        return res[0]
    else:
        name = input(f"🆕 Enter name for new RFID UID {tag_id}: ").strip()
        if name:
            cursor.execute("INSERT INTO rfid_users(tag_id,name) VALUES (?,?)",(str(tag_id),name))
            conn.commit()
            rfid_tags[int(tag_id)] = name
            print(f"✅ Tag saved: UID={tag_id}, Name={name}")
        conn.close()
        return name

init_rfid_db()
rfid_tags = load_rfid_tags()

# ---------------- LOCK CONTROL ----------------
def open_lock(method, identifier):
    global lock_open, current_opener, last_action_time
    with lock_mutex:
        if not lock_open:
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            lock_open = True
            current_opener = f"{method}:{identifier}"
            last_action_time = time.time()
            log_open(method, identifier)

def close_lock(method, identifier):
    global lock_open, current_opener
    with lock_mutex:
        if lock_open and current_opener==f"{method}:{identifier}" and time.time()-last_action_time>=ignore_time:
            GPIO.output(RELAY_PIN, GPIO.LOW)
            lock_open = False
            log_close(method, identifier)
            current_opener=None

# ---------------- RESTORE LOCK STATE ----------------
if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        last_row = df.iloc[-1]
        if last_row['Close Timestamp']=="":
            lock_open=True
            current_opener=f"{last_row['Method']}:{last_row['Identifier']}"
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            print(f"🔓 Lock restored on startup by {current_opener}")
        else:
            GPIO.output(RELAY_PIN, GPIO.LOW)
            print("🔒 Lock CLOSED on startup")
else:
    create_new_csv()
    GPIO.output(RELAY_PIN, GPIO.LOW)
    print("🔒 Lock CLOSED on startup")

# ---------------- FACE THREAD ----------------
def face_access():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,240)
    font = cv2.FONT_HERSHEY_SIMPLEX

    global lock_open, current_opener, last_action_time

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        faces = detector(frame,0)
        for face in faces:
            shape = predictor(frame, face)
            feature = face_rec_model.compute_face_descriptor(frame,shape)
            if known_features:
                distances = [np.linalg.norm(np.array(feature)-np.array(f)) for f in known_features]
                if min(distances)<0.6:
                    name = known_names[distances.index(min(distances))]
                    cv2.putText(frame,name,(face.left(),face.top()-10),font,0.6,(0,255,0),1)
                    open_lock("FACE","-")
                    close_lock("FACE","-")
                else:
                    cv2.putText(frame,"Unknown",(face.left(),face.top()-10),font,0.6,(0,0,255),1)
        cv2.imshow("Face Access",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            break
        time.sleep(0.05)
    cap.release()
    cv2.destroyAllWindows()

# ---------------- RFID THREAD ----------------
def rfid_access():
    global lock_open, current_opener, last_action_time
    while True:
        uid, _ = reader.read_no_block()
        if uid:
            if uid not in rfid_tags:
                name = register_rfid_tag(uid)
            else:
                name = rfid_tags[uid]
                print(f"📟 UID: {uid} | Name: {name}")
            open_lock("RFID", str(uid))
            close_lock("RFID", str(uid))
        time.sleep(0.1)

# ---------------- MAIN ----------------
if __name__=="__main__":
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
