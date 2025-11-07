import sys, os, time, datetime, requests, pandas as pd
from threading import Thread, Lock
import RPi.GPIO as GPIO

# ---------------- CONFIG ----------------
CHANNEL_ID = "3132182"
READ_KEY = "3YJIO2DTT8M2HWJX"
WRITE_KEY = "FTEDV3SYUMLHEUKP"

FIELD_STATUS = 3     # Lock status field
FIELD_CMD = 4        # Command field
UPLOAD_INTERVAL = 20 # ThingSpeak min delay = 15s
CMD_POLL_INTERVAL = 5

LOG_FILE = "access_log.csv"
columns = ["Method", "Identifier", "Open Timestamp", "Close Timestamp"]

# ---------------- GPIO ----------------
GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
RELAY_GPIO = 11
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)  # start locked

csv_mutex = Lock()
lock_open = False
last_status = None

# ---------------- CSV INIT ----------------
if not os.path.exists(LOG_FILE):
    pd.DataFrame(columns=columns).to_csv(LOG_FILE, index=False)

def log_entry(method, identifier, open_time="", close_time=""):
    """Write open/close times into CSV log."""
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
        update_status_to_thingspeak("OPEN", force=True)

def close_lock(method="WEB", identifier="Admin"):
    global lock_open
    if lock_open:
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        lock_open = False
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry(method, identifier, close_time=ts)
        print(f"🔒 Lock closed by {identifier}")
        update_status_to_thingspeak("CLOSED", force=True)

# ---------------- THINGSPEAK ----------------
def update_status_to_thingspeak(status, force=False):
    """Push lock status to Field 3."""
    global last_status
    if status == last_status and not force:
        return
    try:
        r = requests.post("https://api.thingspeak.com/update.json", data={
            "api_key": WRITE_KEY,
            f"field{FIELD_STATUS}": status
        }, timeout=5)

        if r.status_code == 200:
            if r.text.strip() != "0":  # 0 = rate limit hit
                print(f"📡 Field 3 updated → {status}")
                last_status = status
            else:
                print("⚠️ ThingSpeak ignored update (rate limit hit)")
        else:
            print(f"⚠️ Upload failed ({r.status_code})")
    except Exception as e:
        print("⚠️ ThingSpeak update error:", e)

def fetch_command_from_thingspeak():
    """Read command from Field 4."""
    url = f"https://api.thingspeak.com/channels/{CHANNEL_ID}/fields/{FIELD_CMD}/last.json"
    params = {"api_key": READ_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                return str(data.get(f"field{FIELD_CMD}", "")).strip().upper()
    except Exception as e:
        print("⚠️ Command fetch error:", e)
    return ""

def thingspeak_thread():
    """Background thread: listen for commands and push status."""
    last_cmd = ""
    while True:
        # Read latest command
        cmd = fetch_command_from_thingspeak()
        if cmd and cmd != last_cmd:
            print(f"🖥️ Received command: {cmd}")
            last_cmd = cmd
            if cmd == "OPEN_ADMIN":
                open_lock("WEB", "Admin")
            elif cmd == "CLOSE_ADMIN":
                close_lock("WEB", "Admin")

        # Periodically push lock status
        status = "OPEN" if GPIO.input(RELAY_GPIO) == GPIO.HIGH else "CLOSED"
        update_status_to_thingspeak(status)
        time.sleep(UPLOAD_INTERVAL)

# ---------------- MAIN ----------------
if __name__ == "__main__":
    try:
        print("🚀 ThingSpeak Lock Controller started")
        update_status_to_thingspeak("CLOSED", force=True)
        t = Thread(target=thingspeak_thread, daemon=True)
        t.start()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Exiting safely...")
    finally:
        GPIO.output(RELAY_GPIO, GPIO.LOW)
        update_status_to_thingspeak("CLOSED", force=True)
        GPIO.cleanup()
