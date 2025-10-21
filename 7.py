import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import os, cv2, dlib, numpy as np, pandas as pd, datetime, time, sqlite3
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
from threading import Thread, Lock

# ---------------- GPIO SETUP ----------------
RELAY_PIN = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_PIN, GPIO.OUT)
GPIO.output(RELAY_PIN, GPIO.LOW)

# ---------------- CSV LOGGING ----------------
LOG_FILE = "access_log.csv"
log_mutex = Lock()

def create_csv():
    df = pd.DataFrame(columns=["Method","Identifier","Open Timestamp","Close Timestamp","Detected Timestamp"])
    df.to_csv(LOG_FILE, index=False)

if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE)==0:
    create_csv()

def log_to_csv(method, identifier, open_time="", close_time="", detected_time=""):
    with log_mutex:
        df = pd.read_csv(LOG_FILE)
        if open_time or close_time:
            df = pd.concat([df, pd.DataFrame([[method, identifier, open_time, close_time, ""]],
                                             columns=df.columns)])
        elif detected_time:
            df = pd.concat([df, pd.DataFrame([[method, identifier, "", "", detected_time]],
                                             columns=df.columns)])
        df.to_csv(LOG_FILE, index=False)

# ---------------- LOCK STATE ----------------
lock_open = False
current_opener = None
last_action_time = 0
ignore_time = 10  # seconds

def sync_lock_on_startup():
    global lock_open, current_opener
    df = pd.read_csv(LOG_FILE)
    if not df.empty:
        last_open = df[(df["Close Timestamp"]=="") & (df["Open Timestamp"]!="")]
        if not last_open.empty:
            last_row = last_open.iloc[-1]
            lock_open = True
            current_opener = f"{last_row['Method']}:{last_row['Identifier']}"
            GPIO.output(RELAY_PIN, GPIO.HIGH)
            print(f"🔓 Lock opened on startup by {current_opener}")
        else:
            GPIO.output(RELAY_PIN, GPIO.LOW)
            print("🔒 Lock closed on startup")
    else:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        print("🔒 Lock closed on startup")

# ---------------- FACE SETUP ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_model = dlib.face_recognition_model_v1('data/data_dlib/dlib_face_recognition_resnet_model_v1.dat')

def load_face_db():
    names, features = [], []
    if os.path.exists("data/features_all.csv"):
        df = pd.read_csv("data/features_all.csv", header=None)
        for i in range(df.shape[0]):
            names.append(df.iloc[i][0])
            features.append([df.iloc[i][j] for j in range(1,129)])
    return names, features

face_names, face_features = load_face_db()

def euclidean(f1,f2):
    return np.linalg.norm(np.array(f1)-np.array(f2))

# ---------------- RFID SETUP ----------------
reader = SimpleMFRC522()
def init_rfid_db():
    conn = sqlite3.connect("rfid_data.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS rfid_users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tag_id TEXT UNIQUE,
                    name TEXT)''')
    conn.commit()
    conn.close()

def get_or_register_rfid(tag_id):
    conn = sqlite3.connect("rfid_data.db")
    c = conn.cursor()
    c.execute("SELECT name FROM rfid_users WHERE tag_id=?",(str(tag_id),))
    res = c.fetchone()
    if res:
        name = res[0]
    else:
        name = input(f"🆕 Enter name for new RFID {tag_id}: ").strip()
        if name:
            c.execute("INSERT INTO rfid_users(tag_id,name) VALUES (?,?)",(str(tag_id),name))
            conn.commit()
    conn.close()
    if name:
        return name
    else:
        return "Unknown"

# ---------------- LOCK CONTROL ----------------
def open_lock(method, identifier):
    global lock_open, current_opener, last_action_time
    if not lock_open:
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        lock_open = True
        current_opener = f"{method}:{identifier}"
        last_action_time = time.time()
        log_to_csv(method, identifier, open_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"🔓 Lock opened by {current_opener}")

def close_lock(method, identifier):
    global lock_open, current_opener
    if lock_open and current_opener==f"{method}:{identifier}" and time.time()-last_action_time>=ignore_time:
        GPIO.output(RELAY_PIN, GPIO.LOW)
        lock_open = False
        log_to_csv(method, identifier, close_time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        print(f"🔒 Lock closed by {current_opener}")
        current_opener = None

# ---------------- FACE THREAD ----------------
def face_thread():
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
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for f in faces:
            shape = predictor(frame,f)
            feature = face_model.compute_face_descriptor(frame,shape)
            if face_features:
                distances = [euclidean(feature,ff) for ff in face_features]
                if min(distances)<0.6:
                    name = face_names[distances.index(min(distances))]
                    cv2.putText(frame,name,(f.left(),f.top()-10),font,0.6,(0,255,0),1)
                    if not lock_open:
                        open_lock("FACE", name)
                    elif current_opener==f"FACE:{name}" and time.time()-last_action_time>=ignore_time:
                        close_lock("FACE", name)
                else:
                    cv2.putText(frame,"Unknown",(f.left(),f.top()-10),font,0.6,(0,0,255),1)
                    log_to_csv("FACE","Unknown", detected_time=now)
        cv2.imshow("Face Access",frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
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
            now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if name=="Unknown":
                log_to_csv("RFID", f"{uid}(Unknown)", detected_time=now)
            else:
                if not lock_open:
                    open_lock("RFID", f"{name}({uid})")
                elif current_opener==f"RFID:{name}({uid})" and time.time()-last_action_time>=ignore_time:
                    close_lock("RFID", f"{name}({uid})")
        time.sleep(0.1)

# ---------------- MAIN ----------------
if __name__=="__main__":
    try:
        init_rfid_db()
        sync_lock_on_startup()
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
