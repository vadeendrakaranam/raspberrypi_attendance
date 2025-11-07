#!/usr/bin/env python3
# Sentinel Smart Lock Launcher GUI — Fullscreen, Button-based

import tkinter as tk
from tkinter import simpledialog, messagebox
import subprocess
import os

# ---------------- PATHS ----------------
MAIN_PY_PATH = "/home/project/Desktop/Att/1.py"
RFID_PY_PATH = "/home/project/Desktop/Att/rfid.py"
FACE_PY_PATH = "/home/project/Desktop/Att/face.py"

# ---------------- FUNCTIONS ----------------
def kill_script(script_path):
    """Kill all running instances of a given script"""
    try:
        result = subprocess.run(["pgrep", "-f", script_path], capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                os.kill(int(pid), 9)
    except Exception as e:
        print(f"Error killing {script_path}: {e}")

def run_rfid():
    kill_script(MAIN_PY_PATH)
    subprocess.Popen(["python3", RFID_PY_PATH])
    messagebox.showinfo("RFID Module", "RFID system launched successfully!")

def run_face():
    kill_script(MAIN_PY_PATH)
    subprocess.Popen(["python3", FACE_PY_PATH])
    messagebox.showinfo("Face Module", "Face Recognition system launched successfully!")

def close_launcher():
    root.destroy()
    subprocess.Popen(["python3", MAIN_PY_PATH])

# ---------------- GUI ----------------
root = tk.Tk()
root.title("🔒 Sentinel Smart Lock Launcher")
root.attributes("-fullscreen", True)
root.configure(bg="#1E1E2E")  # Dark background

# ---------------- AUTHENTICATION ----------------
def authenticate():
    password = simpledialog.askstring("Authentication", "Enter Password:", show="*")
    if password == "1234":
        show_buttons()
    else:
        messagebox.showerror("Access Denied", "Incorrect Password!")
        root.destroy()

# ---------------- BUTTONS ----------------
def show_buttons():
    label.pack(pady=20)
    subtitle.pack(pady=10)
    rfid_btn.pack(pady=15)
    face_btn.pack(pady=15)
    close_btn.pack(pady=20)

# Styles
button_style = {
    "font": ("Arial", 16, "bold"),
    "bg": "#4ECCA3",
    "fg": "white",
    "activebackground": "#45B38F",
    "activeforeground": "white",
    "relief": "flat",
    "width": 20,
    "height": 2,
    "cursor": "hand2",
}

# Labels
label = tk.Label(root, text="Sentinel Smart Lock", font=("Helvetica", 28, "bold"), fg="#FFD369", bg="#1E1E2E")
subtitle = tk.Label(root, text="Choose Module to Launch", font=("Arial", 14), fg="#BBBBBB", bg="#1E1E2E")

# Buttons
rfid_btn = tk.Button(root, text="🔑 RFID MODULE", command=run_rfid, **button_style)
face_btn = tk.Button(root, text="👁 FACE RECOGNITION", command=run_face, **button_style)
close_btn = tk.Button(root, text="⏹ CLOSE & RETURN", command=close_launcher, **button_style, bg="#FF5555", activebackground="#E04747")

# ---------------- EXIT FULLSCREEN ON ESC ----------------
def exit_fullscreen(event=None):
    root.attributes("-fullscreen", False)
root.bind("<Escape>", exit_fullscreen)

# ---------------- START ----------------
authenticate()  # prompt for password on launch
root.mainloop()
