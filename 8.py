import sqlite3
import RPi.GPIO as GPIO
import time
import datetime
import csv
from mfrc522 import SimpleMFRC522
import os

# --- GPIO setup ---
RELAY_GPIO = 11
GPIO.setmode(GPIO.BOARD)
GPIO.setup(RELAY_GPIO, GPIO.OUT)
GPIO.output(RELAY_GPIO, GPIO.LOW)  # lock initially closed

# --- RFID setup ---
reader = SimpleMFRC522()

# --- Load registered RFID tags ---
def load_registered_rfid_tags():
    tags = {}
    conn = sqlite3.connect("rfid_tags.db")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS rfid_users (id INTEGER PRIMARY KEY AUTOINCREMENT, rfid_id INTEGER UNIQUE, name TEXT)")
    cursor.execute("SELECT rfid_id, name FROM rfid_users")
    rows = cursor.fetchall()
    for rfid_id, name in rows:
        tags[int(rfid_id)] = name
    conn.close()
    return tags

rfid_tags = load_registered_rfid_tags()

CSV_FILE = "rfid_access_log.csv"

# --- CSV helpers ---
def read_csv():
    if not os.path.exists(CSV_FILE):
        return []
    with open(CSV_FILE, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)

def write_csv(rows):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["UID", "Tag Name", "Open Timestamp", "Close Timestamp"])
        writer.writeheader()
        writer.writerows(rows)

def log_rfid_access(uid, tag_name, open_time="", close_time=""):
    rows = read_csv()
    updated = False
    # Check if a row exists for this UID with empty Close Timestamp
    for row in rows:
        if row["UID"] == str(uid) and row["Close Timestamp"] == "":
            if close_time:
                row["Close Timestamp"] = close_time
            updated = True
            break
    if not updated:
        # New entry
        rows.append({
            "UID": str(uid),
            "Tag Name": tag_name,
            "Open Timestamp": open_time,
            "Close Timestamp": close_time
        })
    write_csv(rows)

# --- Track lock state ---
current_access_tag = None
open_time_str = None
other_tags_detected = set()

print("🔑 RFID Lock System Running...")
try:
    while True:
        rfid_id, text = reader.read_no_block()
        if rfid_id:
            if rfid_id in rfid_tags:
                tag_name = rfid_tags[rfid_id]
                now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                if current_access_tag is None:
                    # Lock closed → open
                    GPIO.output(RELAY_GPIO, GPIO.HIGH)
                    current_access_tag = rfid_id
                    open_time_str = now
                    print(f"🔓 Lock opened by {tag_name} at {open_time_str}")
                    log_rfid_access(rfid_id, tag_name, open_time=open_time_str)

                elif current_access_tag == rfid_id:
                    # Same tag closes lock
                    GPIO.output(RELAY_GPIO, GPIO.LOW)
                    close_time_str = now
                    print(f"🔒 Lock closed by {tag_name} at {close_time_str}")
                    log_rfid_access(rfid_id, tag_name, close_time=close_time_str)

                    current_access_tag = None
                    open_time_str = None
                    other_tags_detected.clear()

                else:
                    # Other tag while lock is open → log entry immediately if not already logged
                    if rfid_id not in other_tags_detected:
                        print(f"❌ Other tag detected: {tag_name} at {now}")
                        log_rfid_access(rfid_id, tag_name, open_time=now)
                        other_tags_detected.add(rfid_id)

            else:
                print(f"❌ Unknown RFID tag: {rfid_id}")

            time.sleep(1)
        time.sleep(0.1)

except KeyboardInterrupt:
    print("Exiting...")
finally:
    GPIO.output(RELAY_GPIO, GPIO.LOW)
    GPIO.cleanup()


