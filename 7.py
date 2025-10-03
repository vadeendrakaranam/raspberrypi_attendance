import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import dlib, numpy as np, cv2, pandas as pd, time, sqlite3, datetime
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

# -----------------------------
# Configuration
# -----------------------------
RELAY_GPIO = 11
RELAY_ON, RELAY_OFF = GPIO.HIGH, GPIO.LOW
# How many seconds without seeing the owner before closing the lock
OWNER_TIMEOUT = 1.5   # 1.5 seconds (tweak if needed)
# Minimum confidence threshold for face match
FACE_THRESHOLD = 0.6

# -----------------------------
# Setup hardware / libs
# -----------------------------
GPIO.setmode(GPIO.BOARD)
GPIO.setwarnings(False)         # disable warnings BEFORE setup
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, RELAY_OFF)

reader = SimpleMFRC522()

# Dlib models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# -----------------------------
# DB helpers
# -----------------------------
def load_registered_rfid_tags():
    tags = {}
    conn = sqlite3.connect("rfid_tags.db")
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS rfid_users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        rfid_id INTEGER UNIQUE, name TEXT)""")
    cursor.execute("SELECT rfid_id, name FROM rfid_users")
    for rfid_id, name in cursor.fetchall():
        tags[int(rfid_id)] = name
    conn.close()
    return tags

def ensure_attendance_db():
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("""CREATE TABLE IF NOT EXISTS attendance (
        name TEXT, time TEXT, date DATE, UNIQUE(name, date))""")
    conn.commit()
    conn.close()

def mark_attendance(name):
    current_date = datetime.datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect("attendance.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM attendance WHERE name = ? AND date = ?", (name, current_date))
    if not cursor.fetchone():
        current_time = datetime.datetime.now().strftime('%H:%M:%S')
        cursor.execute("INSERT INTO attendance (name, time, date) VALUES (?, ?, ?)", (name, current_time, current_date))
        conn.commit()
        print(f"✅ {name} marked present at {current_time}")
    conn.close()

# -----------------------------
# Utility
# -----------------------------
def euclidean_distance(f1, f2):
    return np.linalg.norm(np.array(f1) - np.array(f2))

# -----------------------------
# Main App
# -----------------------------
class FaceRFIDSystem:
    def __init__(self, window):
        self.window = window
        self.window.title("Sentinel Smart Lock")
        self.window.geometry("480x800")
        self.window.configure(bg="white")

        # UI
        title_font = ("Arial", 20, "bold")
        time_font = ("Arial", 14)
        self.title_label = tk.Label(window, text="🔒 Sentinel Smart Lock", font=title_font, fg="green", bg="white")
        self.title_label.pack(pady=10)
        self.time_label = tk.Label(window, text="", font=time_font, fg="black", bg="white")
        self.time_label.pack(pady=5)
        self.video_label = tk.Label(window, bg="white")
        self.video_label.pack(pady=20, expand=True)

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 14), padding=10)
        self.reg_button = ttk.Button(window, text="➕ New Registration", command=self.new_registration, style="TButton")
        self.reg_button.pack(pady=20)
        self.quit_button = ttk.Button(window, text="❌ Exit", command=self.on_closing, style="TButton")
        self.quit_button.pack(side="bottom", pady=10)

        # internal
        ensure_attendance_db()
        self.rfid_tags = load_registered_rfid_tags()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.face_features_known_list = []
        self.face_name_known_list = []
        if not self.get_face_database():
            print("❌ Face database not found.")
            # allow running but face will never match
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Lock ownership state:
        # lock_owner_type: None | 'face' | 'rfid'
        # lock_owner_id: for 'face' -> name string, for 'rfid' -> int rfid id
        # lock_owner_last_seen: timestamp of last time owner was seen (for timeout)
        self.lock_owner_type = None
        self.lock_owner_id = None
        self.lock_owner_last_seen = 0.0
        self.lock_open = False

        # Start GUI updates
        self.update_time()
        self.update_frame()
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                features = [csv_rd.iloc[i][j] for j in range(1, 129)]
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                self.face_features_known_list.append(features)
            return True
        return False

    # Lock control helpers
    def open_lock_for_owner(self, owner_type, owner_id, owner_name_for_attendance=None):
        # If lock already open and same owner, just refresh last seen time.
        now = time.time()
        if self.lock_open:
            if self.lock_owner_type == owner_type and self.lock_owner_id == owner_id:
                self.lock_owner_last_seen = now
                return
            else:
                # If lock is open for someone else, ignore new owner request
                return

        # Not currently open → open for this owner
        self.lock_open = True
        self.lock_owner_type = owner_type
        self.lock_owner_id = owner_id
        self.lock_owner_last_seen = now
        GPIO.output(RELAY_GPIO, RELAY_ON)
        print("🔓 Relay ON (owner_type:", owner_type, "owner_id:", owner_id, ")")
        # Attendance mark for people (face) or rfid (use name)
        if owner_type == 'face' and owner_name_for_attendance:
            mark_attendance(owner_name_for_attendance)
        elif owner_type == 'rfid':
            rname = self.rfid_tags.get(owner_id)
            if rname:
                mark_attendance(rname)

    def close_lock_if_timed_out(self):
        if not self.lock_open:
            return
        now = time.time()
        if now - self.lock_owner_last_seen > OWNER_TIMEOUT:
            # close lock
            GPIO.output(RELAY_GPIO, RELAY_OFF)
            print("🔒 Relay OFF (owner timed out)")
            self.lock_open = False
            self.lock_owner_type = None
            self.lock_owner_id = None
            self.lock_owner_last_seen = 0.0

    # Main frame processing
    def process_frame(self, frame):
        owner_seen_in_this_frame = False

        # --- Face detection & recognition ---
        faces = detector(frame, 0)
        for face in faces:
            shape = predictor(frame, face)
            face_feature = face_reco_model.compute_face_descriptor(frame, shape)
            distances = [euclidean_distance(face_feature, f) for f in self.face_features_known_list] if self.face_features_known_list else []
            if distances and min(distances) < FACE_THRESHOLD:
                name = self.face_name_known_list[distances.index(min(distances))]
                # If lock is open for someone else, ignore
                if self.lock_open and not (self.lock_owner_type == 'face' and self.lock_owner_id == name):
                    # ignore this face (someone else holding card or face has opened)
                    cv2.putText(frame, f"(ignored) {name}", (face.left(), face.top() - 10), self.font, 0.6, (0, 0, 255), 1)
                else:
                    # this face can open or keep the lock open
                    self.open_lock_for_owner('face', name, owner_name_for_attendance=name)
                    owner_seen_in_this_frame = True
                    cv2.putText(frame, name, (face.left(), face.top() - 10), self.font, 0.6, (0, 128, 0), 1)
            else:
                name = "unknown"
                cv2.putText(frame, name, (face.left(), face.top() - 10), self.font, 0.6, (0, 0, 0), 1)

            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 0), 2)

        # --- RFID detection (non-blocking) ---
        try:
            # Prefer read_id_no_block if available; fallback to read_no_block
            if hasattr(reader, "read_id_no_block"):
                rfid_id = reader.read_id_no_block()
            else:
                # some versions expose read_no_block which returns (id, text) or raises when no card
                try:
                    tmp = reader.read_no_block()
                    # if it returns a tuple or single int, handle accordingly
                    if isinstance(tmp, tuple):
                        rfid_id = tmp[0]
                    else:
                        rfid_id = tmp
                except Exception:
                    rfid_id = None
        except Exception:
            rfid_id = None

        if rfid_id:
            rfid_name = self.rfid_tags.get(int(rfid_id))
            if rfid_name:
                # If lock is open for someone else, ignore
                if self.lock_open and not (self.lock_owner_type == 'rfid' and self.lock_owner_id == int(rfid_id)):
                    # ignore
                    cv2.putText(frame, f"(ignored RFID)", (10, 30), self.font, 0.7, (0, 0, 255), 2)
                else:
                    # open or refresh lock for this RFID
                    self.open_lock_for_owner('rfid', int(rfid_id))
                    owner_seen_in_this_frame = True
                    cv2.putText(frame, f"RFID: {rfid_name}", (10, 30), self.font, 0.7, (0, 128, 0), 2)

        # If owner seen this frame, update last seen (already updated inside open_lock_for_owner)
        # If not seen, attempt to check proximity for current owner:
        if not owner_seen_in_this_frame and self.lock_open:
            # if the owner_type is 'face', we relied on face detection per frame (so absence means not seen)
            # if the owner_type is 'rfid', we need to actively check if that RFID is still present
            if self.lock_owner_type == 'rfid':
                # check if same card still present this frame
                still_present = False
                try:
                    if hasattr(reader, "read_id_no_block"):
                        rid = reader.read_id_no_block()
                        if rid and int(rid) == int(self.lock_owner_id):
                            still_present = True
                    else:
                        try:
                            tmp = reader.read_no_block()
                            rid = tmp[0] if isinstance(tmp, tuple) else tmp
                            if rid and int(rid) == int(self.lock_owner_id):
                                still_present = True
                        except Exception:
                            still_present = False
                except Exception:
                    still_present = False

                if still_present:
                    self.lock_owner_last_seen = time.time()
                # else leave last_seen alone and let timeout close the lock

        # Close lock if timed out
        self.close_lock_if_timed_out()

        return frame

    # GUI updates
    def update_time(self):
        now = datetime.datetime.now().strftime("%I:%M %p | %b %d, %Y")
        self.time_label.config(text=now)
        self.window.after(1000, self.update_time)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            frame = self.process_frame(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.rotate(frame, cv2.ROTATE_180)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
        self.window.after(20, self.update_frame)

    def new_registration(self):
        # Example: launch external registration script if you have one
        print("👉 New Registration pressed")
        # os.system("python3 register.py")

    def on_closing(self):
        self.cap.release()
        cv2.destroyAllWindows()
        GPIO.cleanup()
        self.window.destroy()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRFIDSystem(root)
    root.mainloop()
