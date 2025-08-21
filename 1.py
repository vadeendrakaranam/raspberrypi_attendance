import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import dlib, cv2, numpy as np, pandas as pd, time, sqlite3, datetime
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
import tkinter as tk
from PIL import Image, ImageTk

# --- Setup GPIO ---
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)

reader = SimpleMFRC522()

# --- Load Face Models ---
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("data/data_dlib/shape_predictor_68_face_landmarks.dat")
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# --- DB Setup ---
def load_registered_rfid_tags():
    tags = {}
    conn = sqlite3.connect("authorized.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS rfid_users (id INTEGER PRIMARY KEY AUTOINCREMENT, rfid_id INTEGER UNIQUE, name TEXT)")
    cursor.execute("SELECT rfid_id, name FROM rfid_users")
    for rfid_id, name in cursor.fetchall():
        tags[int(rfid_id)] = name
    conn.close()
    return tags

class SentinelApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Sentinel Smart Lock")
        self.root.configure(bg="white")

        # Fonts/colors
        GREEN = "#1B8E3F"
        FONT_HEADER = ("Arial", 22, "bold")

        # --- Header ---
        self.header = tk.Label(root, text="🔒 Sentinel Smart Lock", font=FONT_HEADER, fg=GREEN, bg="white")
        self.header.pack(pady=10)

        # --- Time ---
        self.time_label = tk.Label(root, font=("Arial", 14), fg=GREEN, bg="white")
        self.time_label.pack()

        # --- Camera Frame ---
        self.video_label = tk.Label(root, bg="#111111", width=320, height=240)
        self.video_label.pack(pady=15)

        # --- Button ---
        self.btn_register = tk.Button(root, text="New Registration ➜", bg=GREEN, fg="white",
                                      font=("Arial", 14), command=self.new_registration)
        self.btn_register.pack(pady=10)

        # --- Status ---
        self.status = tk.Label(root, text="● System Online", fg=GREEN, bg="white", font=("Arial", 12))
        self.status.pack(side="bottom", pady=10)

        # --- Face/RFID Logic ---
        self.rfid_tags = load_registered_rfid_tags()
        self.cap = cv2.VideoCapture(0)
        self.face_features_known_list, self.face_name_known_list = self.load_face_db()
        self.relay_on = False
        self.relay_time = 0

        self.update_time()
        self.update_frame()

    def load_face_db(self):
        faces, names = [], []
        if os.path.exists("data/features_all.csv"):
            csv_rd = pd.read_csv("data/features_all.csv", header=None)
            for i in range(csv_rd.shape[0]):
                names.append(csv_rd.iloc[i][0])
                faces.append([csv_rd.iloc[i][j] for j in range(1, 129)])
        return faces, names

    def update_time(self):
        now = datetime.datetime.now().strftime("%I:%M %p, %b %d, %Y")
        self.time_label.config(text=now)
        self.root.after(1000, self.update_time)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            known_detected = False
            faces = detector(frame, 0)
            for face in faces:
                shape = predictor(frame, face)
                face_feature = face_reco_model.compute_face_descriptor(frame, shape)
                distances = [np.linalg.norm(np.array(face_feature) - np.array(f)) for f in self.face_features_known_list]
                if distances and min(distances) < 0.6:
                    name = self.face_name_known_list[distances.index(min(distances))]
                    cv2.putText(frame, name, (face.left(), face.top()-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                    known_detected = True
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0,255,0), 2)

            # RFID check
            try:
                id, _ = reader.read_no_block()
                if id and id in self.rfid_tags:
                    rfid_name = self.rfid_tags[id]
                    cv2.putText(frame, f"RFID: {rfid_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    known_detected = True
            except:
                pass

            # Relay
            now = time.time()
            if known_detected and not self.relay_on:
                GPIO.output(RELAY_GPIO, GPIO.HIGH)
                self.relay_on = True
                self.relay_time = now
            if self.relay_on and now - self.relay_time > 10:
                GPIO.output(RELAY_GPIO, GPIO.LOW)
                self.relay_on = False

            # Show frame in Tkinter
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imgtk = ImageTk.PhotoImage(image=Image.fromarray(frame_rgb))
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)

        self.root.after(30, self.update_frame)

    def new_registration(self):
        os.system("python3 register_rfid.py")

if __name__ == "__main__":
    root = tk.Tk()
    app = SentinelApp(root)
    root.mainloop()
