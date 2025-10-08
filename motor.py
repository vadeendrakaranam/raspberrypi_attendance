import sys, os
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

import dlib, cv2, numpy as np, pandas as pd, sqlite3, time, datetime, csv
import tkinter as tk
from threading import Thread, Lock
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522

# ---------------- GPIO Setup ----------------
# L298N Motor pins
ENA, IN1, IN2 = 12, 5, 6   # Motor 1
ENB, IN3, IN4 = 13, 20, 21 # Motor 2

GPIO.setmode(GPIO.BCM)
for pin in [ENA, IN1, IN2, ENB, IN3, IN4]:
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.LOW)

pwmA = GPIO.PWM(ENA, 1000)
pwmB = GPIO.PWM(ENB, 1000)
pwmA.start(0)
pwmB.start(0)

# ---------------- Global State ----------------
auth_state = {
    "face_verified": False,
    "rfid_verified": False,
    "operator_name": None,
    "rfid_id": None
}
auth_lock = Lock()

# ---------------- Load Face Database ----------------
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')
face_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

face_names = []
face_features = []

if os.path.exists("data/features_all.csv"):
    df = pd.read_csv("data/features_all.csv", header=None)
    for i in range(df.shape[0]):
        face_names.append(df.iloc[i][0])
        face_features.append([df.iloc[i][j] for j in range(1, 129)])
else:
    print("❌ Face database missing!")
    exit()

# ---------------- Load RFID DB ----------------
conn = sqlite3.connect("rfid_tags.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS rfid_users (id INTEGER PRIMARY KEY AUTOINCREMENT, rfid_id INTEGER UNIQUE, name TEXT)")
c.execute("SELECT rfid_id, name FROM rfid_users")
rfid_to_name = {str(row[0]): row[1] for row in c.fetchall()}
conn.close()

# ---------------- CSV Logging ----------------
LOG_FILE = "access_log.csv"

def log_access(name, open_time, close_time):
    date_str = open_time.split(" ")[0]
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Name", "Date", "Open Time", "Close Time"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"Name": name, "Date": date_str, "Open Time": open_time, "Close Time": close_time})

# ---------------- Face Detection Thread ----------------
class FaceThread(Thread):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.cap = cv2.VideoCapture(0)
    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            faces = face_detector(frame, 0)
            for face in faces:
                shape = predictor(frame, face)
                feature = face_model.compute_face_descriptor(frame, shape)
                distances = [np.linalg.norm(np.array(feature)-np.array(f)) for f in face_features]
                if distances and min(distances) < 0.6:
                    name = face_names[distances.index(min(distances))]
                    with auth_lock:
                        auth_state["face_verified"] = True
                        auth_state["operator_name"] = name
                    print(f"✅ Face verified: {name}")
                    self.callback()
                    self.cap.release()
                    cv2.destroyAllWindows()
                    return
            time.sleep(0.05)

# ---------------- RFID Detection Thread ----------------
class RFIDThread(Thread):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.reader = SimpleMFRC522()
    def run(self):
        while True:
            rfid_id, _ = self.reader.read_no_block()
            if rfid_id:
                rfid_str = str(rfid_id)
                if rfid_str in rfid_to_name:
                    with auth_lock:
                        auth_state["rfid_verified"] = True
                        auth_state["rfid_id"] = rfid_str
                        auth_state["operator_name"] = rfid_to_name[rfid_str]
                    print(f"✅ RFID verified: {rfid_to_name[rfid_str]}")
                    self.callback()
                    return
            time.sleep(0.1)

# ---------------- Motor Control Functions ----------------
def motorA_forward(speed=80):
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    pwmA.ChangeDutyCycle(speed)

def motorA_stop():
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.LOW)
    pwmA.ChangeDutyCycle(0)

def motorB_forward(speed=80):
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)
    pwmB.ChangeDutyCycle(speed)

def motorB_stop():
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.LOW)
    pwmB.ChangeDutyCycle(0)

# ---------------- Motor Control GUI ----------------
class MotorGUI:
    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("Motor Control")
        self.win.geometry("350x250")
        self.open_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.operator_name = auth_state["operator_name"]
        self.create_widgets()

    def create_widgets(self):
        tk.Label(self.win, text=f"Operator: {self.operator_name}", font=("Arial", 12, "bold")).pack(pady=10)
        # Motor 1
        tk.Button(self.win, text="Motor 1 ON", command=lambda: motorA_forward()).pack(pady=5)
        tk.Button(self.win, text="Motor 1 OFF", command=lambda: motorA_stop()).pack(pady=5)
        # Motor 2
        tk.Button(self.win, text="Motor 2 ON", command=lambda: motorB_forward()).pack(pady=5)
        tk.Button(self.win, text="Motor 2 OFF", command=lambda: motorB_stop()).pack(pady=5)
        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

    def show(self):
        self.win.grab_set()
        self.win.wait_window()

    def on_close(self):
        motorA_stop()
        motorB_stop()
        close_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_access(self.operator_name, self.open_time, close_time)
        with auth_lock:
            auth_state["face_verified"] = False
            auth_state["rfid_verified"] = False
            auth_state["rfid_id"] = None
            auth_state["operator_name"] = None
        self.win.destroy()

# ---------------- Home GUI ----------------
class HomeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Home - Access Control")
        self.root.geometry("400x300")
        self.face_tick = tk.StringVar(value="❌")
        self.rfid_tick = tk.StringVar(value="❌")
        self.create_widgets()
        self.update_time()
        self.root.mainloop()

    def create_widgets(self):
        self.time_label = tk.Label(self.root, font=("Arial", 14))
        self.time_label.pack(pady=10)

        self.face_btn = tk.Button(self.root, text="Face Detection", width=20, command=self.start_face_detection)
        self.face_btn.pack(pady=5)
        tk.Label(self.root, textvariable=self.face_tick, font=("Arial", 14)).pack()

        self.rfid_btn = tk.Button(self.root, text="RFID Detection", width=20, command=self.start_rfid_detection)
        self.rfid_btn.pack(pady=5)
        tk.Label(self.root, textvariable=self.rfid_tick, font=("Arial", 14)).pack()

    def update_time(self):
        self.time_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.root.after(1000, self.update_time)

    def start_face_detection(self):
        self.face_tick.set("⏳")
        FaceThread(callback=self.face_detected).start()

    def start_rfid_detection(self):
        self.rfid_tick.set("⏳")
        RFIDThread(callback=self.rfid_detected).start()

    def face_detected(self):
        self.face_tick.set("✅")
        self.check_both_verified()

    def rfid_detected(self):
        self.rfid_tick.set("✅")
        self.check_both_verified()
      
    def check_both_verified(self):
        with auth_lock:
            # Check if both face and RFID verified and names match
            if auth_state["face_verified"] and auth_state["rfid_verified"]:
                if rfid_to_name.get(auth_state["rfid_id"]) == auth_state["operator_name"]:
                    # Open Motor Control GUI
                    MotorGUI(self.root).show()
                    # Reset ticks after closing Motor GUI
                    self.face_tick.set("❌")
                    self.rfid_tick.set("❌")
                else:
                    print(f"❌ Face-RFID mismatch: {auth_state['operator_name']} vs {rfid_to_name.get(auth_state['rfid_id'])}")
                    tk.messagebox.showerror("Mismatch", "Face and RFID do not match!")
                    # Reset only RFID to retry
                    self.rfid_tick.set("❌")
                    auth_state["rfid_verified"] = False
                    auth_state["rfid_id"] = None

# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        HomeGUI()
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        motorA_stop()
        motorB_stop()
        GPIO.cleanup()
