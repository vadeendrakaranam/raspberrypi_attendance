#!/usr/bin/env python3
# Fullscreen Sentinel Smart Lock Launcher with Embedded Authentication

import tkinter as tk
from tkinter import messagebox
import subprocess
import os
import signal

# ------------------------ PATHS ------------------------
MAIN_PY_PATH = "/home/project/Desktop/Att/1.py"
RFID_PY_PATH = "/home/project/Desktop/Att/rfid.py"
FACE_PY_PATH = "/home/project/Desktop/Att/face.py"

# ------------------------ UTILITIES ------------------------
def kill_script(script_path):
    """Kill all running instances of a given script"""
    try:
        result = subprocess.run(["pgrep", "-f", script_path], capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                os.kill(int(pid), signal.SIGKILL)
                print(f"✅ Killed {script_path} PID: {pid}")
    except Exception as e:
        print(f"❌ Error killing {script_path}: {e}")

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

# ------------------------ GUI SETUP ------------------------
root = tk.Tk()
root.title("🔒 Sentinel Smart Lock Launcher")
root.attributes("-fullscreen", True)
root.configure(bg="#1E1E2E")  # dark background

# Fullscreen toggle with ESC
fullscreen = True
def toggle_fullscreen(event=None):
    global fullscreen
    fullscreen = not fullscreen
    root.attributes("-fullscreen", fullscreen)
root.bind("<Escape>", toggle_fullscreen)

# ------------------------ STYLES ------------------------
button_style = {
    "font": ("Arial", 16, "bold"),
    "bg": "#4ECCA3",
    "fg": "white",
    "activebackground": "#45B38F",
    "activeforeground": "white",
    "relief": "flat",
    "width": 20,
    "height": 2,
    "bd": 0,
    "cursor": "hand2",
}

# ------------------------ LABELS ------------------------
label = tk.Label(
    root,
    text="Sentinel Smart Lock",
    font=("Helvetica", 40, "bold"),
    fg="#FFD369",
    bg="#1E1E2E",
)
label.pack(pady=50)

subtitle = tk.Label(
    root,
    text="Enter Password to Continue",
    font=("Arial", 24),
    fg="#BBBBBB",
    bg="#1E1E2E",
)
subtitle.pack(pady=20)

# ------------------------ PASSWORD ENTRY ------------------------
password_var = tk.StringVar()

password_entry = tk.Entry(root, textvariable=password_var, show="*", font=("Arial", 20))
password_entry.pack(pady=20)
password_entry.focus_set()

def authenticate():
    if password_var.get() == "1234":
        password_frame.pack_forget()
        show_buttons()
    else:
        messagebox.showerror("Access Denied", "Incorrect Password!")
        password_var.set("")
        password_entry.focus_set()

# ------------------------ AUTH FRAME ------------------------
password_frame = tk.Frame(root, bg="#1E1E2E")
password_frame.pack()
password_frame.pack_propagate(False)

password_button = tk.Button(password_frame, text="🔒 Unlock", command=authenticate, **button_style)
password_button.pack(pady=10)

# ------------------------ MODULE BUTTONS ------------------------
buttons_frame = tk.Frame(root, bg="#1E1E2E")
buttons_frame.pack(pady=50)

def show_buttons():
    # Clear password widgets
    password_frame.pack_forget()
    
    # Buttons
    rfid_btn = tk.Button(buttons_frame, text="🔑 RFID MODULE", command=run_rfid, **button_style)
    rfid_btn.pack(pady=20)
    
    face_btn = tk.Button(buttons_frame, text="👁 FACE RECOGNITION", command=run_face, **button_style)
    face_btn.pack(pady=20)
    
    close_btn = tk.Button(buttons_frame, text="⏹ CLOSE & RETURN", command=close_launcher, **button_style)
    close_btn.config(bg="#FF5555", activebackground="#E04747")
    close_btn.pack(pady=30)

# ------------------------ FOOTER ------------------------
footer = tk.Label(root, text="Developed by Vadeendra Karanam",
                  font=("Helvetica", 16), bg="#1E1E2E", fg="#BBBBBB")
footer.pack(side="bottom", pady=20)

# ------------------------ RUN LOOP ------------------------
root.mainloop()
