# rpi_lock_thingspeak.py
import RPi.GPIO as GPIO
import time, datetime, requests, pandas as pd, os, json
from threading import Thread, Lock

# ---------------- CONFIG ----------------
CHANNEL_ID = "3132182"
READ_KEY = "3YJIO2DTT8M2HWJX"
WRITE_KEY = "FTEDV3SYUMLHEUKP"

FIELD_STATUS = 3    # lock status field
FIELD_CMD = 4       # command field
UPLOAD_INTERVAL = 10  # seconds
CMD_POLL_INTERVAL = 5

LOG_FILE = "access_log.csv"
columns = ["Method", "Identifier", "Open Timestamp", "Close Timestamp"]

# ---------------- GPIO ----------------
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
RELAY_GPIO = 11
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)   # initially locked

csv_mutex = Lock()
lock_open = False

# ---------------- CSV INIT ----------------
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)

def log_entry(method, identifier, open_time="", close_time=""):
    with csv_mutex:
        df = pd.read_csv(LOG_FILE)
        if open_time:
            df.loc[len(df)] = [method, identifier, open_time, ""]
        elif close_time:
            mask = (df["Close Timestamp"] == "") & (df["Open Timestamp"] != "")
            if mask.any():
                idx = df[mask].index[-1]
                df.loc[idx, "Close Timestamp"] = close_time
            else:
                df.loc[len(df)] = [method, identifier, "", close_time]
        df.to_csv(LOG_FILE, index=False)

# ---------------- LOCK CONTROL ----------------
def open_lock(method="WEB", identifier="Admin"):
    global lock_open
    if not lock_open:
        GPIO.output(RELAY_GPIO, GPIO.HIGH)
        lock_open = True
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry(method, identifier, open_time=ts)
        print(f"🔓 Lock opened by {identifier}")
        update_status_to_thingspeak("OPEN")

def close_lock(method="WEB", identifier="Admin"):
    global lock_open
    if lock_open:
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        lock_open = False
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry(method, identifier, close_time=ts)
        print(f"🔒 Lock closed by {identifier}")
        update_status_to_thingspeak("CLOSED")

# ---------------- THINGSPEAK SYNC ----------------
def update_status_to_thingspeak(status):
    """Push current lock status to Field 3."""
    try:
        r = requests.post("https://api.thingspeak.com/update.json", data={
            "api_key": WRITE_KEY,
            f"field{FIELD_STATUS}": status
        }, timeout=10)
        if r.status_code == 200:
            print(f"📡 Status uploaded: {status}")
    except Exception as e:
        print("⚠️ ThingSpeak update error:", e)

def fetch_command_from_thingspeak():
    """Read latest command from Field 4."""
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/fields/{FIELD_CMD}/last.json"
    params = {"api_key": READ_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            cmd = str(data.get(f"field{FIELD_CMD}", "")).strip().upper()
            return cmd
    except Exception as e:
        print("⚠️ Command fetch error:", e)
    return ""

def thingspeak_thread():
    last_cmd = ""
    while True:
        cmd = fetch_command_from_thingspeak()
        if cmd and cmd != last_cmd:
            print(f"🖥️ Received command: {cmd}")
            last_cmd = cmd
            if cmd == "OPEN_ADMIN":
                open_lock("WEB", "Admin")
            elif cmd == "CLOSE_ADMIN":
                close_lock("WEB", "Admin")
        # update current GPIO status periodically
        status = "OPEN" if GPIO.input(RELAY_GPIO) == GPIO.HIGH else "CLOSED"
        update_status_to_thingspeak(status)
        time.sleep(UPLOAD_INTERVAL)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        print("🚀 ThingSpeak Lock Controller started")
        update_status_to_thingspeak("CLOSED")
        t = Thread(target=thingspeak_thread, daemon=True)
        t.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("🛑 Exiting...")
    finally:
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        update_status_to_thingspeak("CLOSED")
        GPIO.cleanup()

