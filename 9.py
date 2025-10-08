import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import dlib
import numpy as np
import cv2
import pandas as pd
import time
import datetime
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from threading import Thread, Lock

# ---------------- GPIO SETUP ----------------
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)

# ---------------- LOCK STATE FILE ----------------
LOCK_STATE_FILE = "lock_state.txt"
lock_file_mutex = Lock()
def get_lock_state():
    with lock_file_mutex:
        if not os.path.exists(LOCK_STATE_FILE):
            return {"status":"CLOSED","system":None}
        with open(LOCK_STATE_FILE,"r") as f:
            lines = f.readlines()
        state = {}
        for line in lines:
            if "=" in line:
                k,v = line.strip().split("=")
                state[k]=v
        return state

def set_lock_state(status, system):
    with lock_file_mutex:
        with open(LOCK_STATE_FILE,"w") as f:
            f.write(f"status={status}\n")
            f.write(f"system={system}\n")
            f.write(f"timestamp={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.flush()
            os.fsync(f.fileno())

# ---------------- CSV LOGGING ----------------
FACE_CSV = "face_access_log.csv"
RFID_CSV = "rfid_access_log.csv"

def log_face(name, open_time="", close_time=""):
    rows=[]
    if os.path.exists(FACE_CSV):
        df=pd.read_csv(FACE_CSV)
        rows=df.to_dict("records")
    updated=False
    for row in rows:
        if row["Name"]==name and row["Close Timestamp"]=="":
            if close_time: row["Close Timestamp"]=close_time
            updated=True
            break
    if not updated:
        rows.append({"Name":name,"Open Timestamp":open_time,"Close Timestamp":close_time})
    pd.DataFrame(rows).to_csv(FACE_CSV,index=False)

def log_rfid(uid, tag_name, open_time="", close_time=""):
    rows=[]
    if os.path.exists(RFID_CSV):
        df=pd.read_csv(RFID_CSV)
        rows=df.to_dict("records")
    updated=False
    for row in rows:
        if row["UID"]==str(uid) and row["Close Timestamp"]=="":
            if close_time: row["Close Timestamp"]=close_time
            updated=True
            break
    if not updated:
        rows.append({"UID":str(uid),"Tag Name":tag_name,"Open Timestamp":open_time,"Close Timestamp":close_time})
    pd.DataFrame(rows).to_csv(RFID_CSV,index=False)

# ---------------- DLIB SETUP ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# ---------------- GLOBAL STATE ----------------
auth_state = {"face_verified":False,"rfid_verified":False,"name":None}

# ---------------- FACE DETECTION THREAD ----------------
class FaceThread(Thread):
    def __init__(self, face_label, status_label):
        super().__init__()
        self.face_label = face_label
        self.status_label = status_label
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        self.face_features_known_list=[]
        self.face_name_known_list=[]
        self.load_face_db()
        self.current_user=None
        self.lock_open_time=0
        self.ignore_duration=10
        self.last_closed_time={}

    def load_face_db(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd=pd.read_csv("data/features_all.csv",header=None)
            for i in range(csv_rd.shape[0]):
                features=[csv_rd.iloc[i][j] for j in range(1,129)]
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_features_known_list.append(features)
        else:
            print("Face database not found")
            exit()

    def euclidean_distance(self,f1,f2):
        return np.linalg.norm(np.array(f1)-np.array(f2))

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            faces = detector(frame, 0)
            for face in faces:
                shape = predictor(frame, face)
                feature = face_reco_model.compute_face_descriptor(frame, shape)
                distances = [self.euclidean_distance(feature,f) for f in self.face_features_known_list]
                if distances and min(distances)<0.6:
                    name = self.face_name_known_list[distances.index(min(distances))]
                    auth_state["face_verified"]=True
                    auth_state["name"]=name
                    self.status_label.config(text=f"✅ Face Verified: {name}")
                    now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_face(name, open_time=now)
                else:
                    auth_state["face_verified"]=False
                    self.status_label.config(text="❌ Unknown Face")
            # Convert to Tkinter image
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.face_label.imgtk=imgtk
            self.face_label.configure(image=imgtk)
            time.sleep(0.03)

# ---------------- RFID DETECTION THREAD ----------------
class RFIDThread(Thread):
    def __init__(self, status_label):
        super().__init__()
        self.reader = SimpleMFRC522()
        self.status_label = status_label
        self.rfid_tags=self.load_tags()
        self.current_tag=None
        self.ignore_duration=10
        self.last_closed_time={}

    def load_tags(self):
        tags={}
        conn=sqlite3.connect("rfid_tags.db")
        cursor=conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS rfid_users (id INTEGER PRIMARY KEY AUTOINCREMENT, rfid_id INTEGER UNIQUE, name TEXT)")
        cursor.execute("SELECT rfid_id,name FROM rfid_users")
        rows=cursor.fetchall()
        for rfid_id,name in rows:
            tags[int(rfid_id)]=name
        conn.close()
        return tags

    def run(self):
        while True:
            self.status_label.config(text="Reading RFID...")
            rfid_id, text = self.reader.read()  # blocking read
            if rfid_id:
                tag_name=self.rfid_tags.get(rfid_id,str(rfid_id))
                now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if auth_state["face_verified"] and auth_state["name"]==tag_name:
                    auth_state["rfid_verified"]=True
                    self.status_label.config(text=f"✅ RFID Verified: {tag_name}")
                    log_rfid(rfid_id, tag_name, open_time=now)
                else:
                    auth_state["rfid_verified"]=False
                    self.status_label.config(text=f"❌ RFID mismatch or unknown")
            time.sleep(0.1)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("Access Control System")
root.geometry("700x400")

# Navbar
navbar = tk.Label(root,text="",bg="lightgrey",font=("Arial",12))
navbar.pack(fill=tk.X)
def update_time():
    now=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    navbar.config(text=f"Current Date & Time: {now}")
    root.after(500,update_time)
update_time()

# Main frames
main_frame=tk.Frame(root)
main_frame.pack(fill=tk.BOTH,expand=True,padx=10,pady=10)

# Left: Face Detection
left_frame=tk.LabelFrame(main_frame,text="Face Detection",width=350,height=350)
left_frame.pack(side=tk.LEFT,fill=tk.BOTH,expand=True,padx=5,pady=5)
face_label=tk.Label(left_frame,width=40,height=15,bg="black")
face_label.pack(pady=5)
face_status_label=tk.Label(left_frame,text="Not Started")
face_status_label.pack(pady=5)
def start_face_thread():
    ft=FaceThread(face_label,face_status_label)
    ft.daemon=True
    ft.start()
tk.Button(left_frame,text="Start Face Detection",command=start_face_thread).pack(pady=5)

# Right: RFID Detection
right_frame=tk.LabelFrame(main_frame,text="RFID Detection",width=300,height=300)
right_frame.pack(side=tk.RIGHT,fill=tk.BOTH,expand=True,padx=5,pady=5)
rfid_status_label=tk.Label(right_frame,text="Not Started")
rfid_status_label.pack(pady=20)
def start_rfid_thread():
    rt=RFIDThread(rfid_status_label)
    rt.daemon=True
    rt.start()
tk.Button(right_frame,text="Start RFID Detection",command=start_rfid_thread).pack(pady=5)

# Relay control
def open_relay_gui():
    relay_win=tk.Toplevel()
    relay_win.title("Relay Control Panel")
    relay_win.geometry("300x200")
    relay_win.grab_set()
    tk.Label(relay_win,text=f"User: {auth_state['name']}",font=("Arial",12,"bold")).pack(pady=10)
    tk.Label(relay_win,text="Relay is ON",font=("Arial",12,"bold"),fg="green").pack(pady=20)
    # When closed, reset auth
    def on_close():
        auth_state["face_verified"]=False
        auth_state["rfid_verified"]=False
        auth_state["name"]=None
        GPIO.output(RELAY_GPIO,GPIO.LOW)
        relay_win.destroy()
    relay_win.protocol("WM_DELETE_WINDOW",on_close)

tk.Button(root,text="Open Relay Control",bg="green",fg="white",command=open_relay_gui).pack(pady=10)

root.mainloop()

GPIO.output(RELAY_GPIO,GPIO.LOW)
GPIO.cleanup()
