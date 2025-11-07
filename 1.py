#!/usr/bin/env python3
# launcher_gui.py — Sentinel Smart Lock Launcher (RFID / FACE)

import tkinter as tk
from tkinter import simpledialog, messagebox
import subprocess
import os
import signal
import sys

MAIN_PY_PATH = "/home/project/Desktop/Att/1.py"
RFID_PY_PATH = "/home/project/Desktop/Att/rfid.py"
FACE_PY_PATH = "/home/project/Desktop/Att/face.py"

# ------------------------ Utility Functions ------------------------

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
    root.destroy()

def run_face():
    kill_script(MAIN_PY_PATH)
    subprocess.Popen(["python3", FACE_PY_PATH])
    root.destroy()

def close_launcher():
    root.destroy()
    subprocess.Popen(["python3", MAIN_PY_PATH])

# ------------------------ Password Prompt ------------------------
root_pw = tk.Tk()
root_pw.withdraw()  # hide small main window
password = simpledialog.askstring("🔒 Password Required", "Enter Admin Password:", show="*")

if password != "1234":
    messagebox.showerror("Access Denied", "Incorrect Password!")
    sys.exit(0)

root_pw.destroy()

# ------------------------ Main Launcher GUI ------------------------
root = tk.Tk()
root.title("Sentinel Smart Lock — Launcher")
root.attributes("-fullscreen", True)

# Colors & fonts
BG_COLOR = "#081A2B"
BTN_COLOR = "#1E7A6F"
BTN_HOVER = "#289B84"
TEXT_COLOR = "#A3FFD9"
FOOTER_COLOR = "#99B7C2"

root.configure(bg=BG_COLOR)

# Header
header = tk.Label(
    root,
    text="🛡️ Sentinel Smart Lock",
    font=("Helvetica", 42, "bold"),
    bg=BG_COLOR,
    fg="#7BE0E0"
)
header.pack(pady=(60, 20))

sub_header = tk.Label(
    root,
    text="Launcher Dashboard",
    font=("Helvetica", 24),
    bg=BG_COLOR,
    fg=TEXT_COLOR
)
sub_header.pack(pady=(0, 40))

# Button hover effects
def on_enter(e): e.widget.config(bg=BTN_HOVER)
def on_leave(e): e.widget.config(bg=BTN_COLOR)

# Buttons Frame
btn_frame = tk.Frame(root, bg=BG_COLOR)
btn_frame.pack(pady=20)

btn_rfid = tk.Button(
    btn_frame,
    text="📡  RFID MODULE",
    font=("Helvetica", 22, "bold"),
    bg=BTN_COLOR,
    fg="white",
    width=22,
    height=2,
    command=run_rfid,
    relief="flat",
    bd=0,
)
btn_rfid.pack(pady=15)
btn_rfid.bind("<Enter>", on_enter)
btn_rfid.bind("<Leave>", on_leave)

btn_face = tk.Button(
    btn_frame,
    text="🙂  FACE MODULE",
    font=("Helvetica", 22, "bold"),
    bg=BTN_COLOR,
    fg="white",
    width=22,
    height=2,
    command=run_face,
    relief="flat",
    bd=0,
)
btn_face.pack(pady=15)
btn_face.bind("<Enter>", on_enter)
btn_face.bind("<Leave>", on_leave)

btn_close = tk.Button(
    root,
    text="⬅️  RETURN TO MAIN",
    font=("Helvetica", 20, "bold"),
    bg="#244A57",
    fg="white",
    width=20,
    height=1,
    command=close_launcher,
    relief="flat",
    bd=0,
)
btn_close.pack(pady=(60, 10))

# Footer
footer = tk.Label(
    root,
    text="Developed by Vadeendra Karanam",
    font=("Helvetica", 16),
    bg=BG_COLOR,
    fg=FOOTER_COLOR
)
footer.pack(side="bottom", pady=20)

# Exit with Esc
root.bind("<Escape>", lambda e: root.destroy())

root.mainloop()
