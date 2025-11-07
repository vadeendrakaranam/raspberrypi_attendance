#!/usr/bin/env python3
# Fullscreen Smart Lock Launcher with Password

import tkinter as tk
from tkinter import messagebox
import subprocess
import os

# ------------------------ PATHS ------------------------
MAIN_PY_PATH = "/home/project/Desktop/Att/1.py"
RFID_PY_PATH = "/home/project/Desktop/Att/rfid.py"
FACE_PY_PATH = "/home/project/Desktop/Att/face.py"

# ------------------------ MODULE LAUNCHERS ------------------------
def run_module(path, name):
    try:
        subprocess.Popen(["python3", path])
        messagebox.showinfo(f"{name}", f"{name} launched successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch {name}:\n{e}")

# ------------------------ GUI ------------------------
class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔒 Sentinel Smart Lock Launcher")
        self.root.configure(bg="#1E1E2E")

        # Force fullscreen immediately
        root.update_idletasks()
        root.deiconify()
        root.attributes("-fullscreen", True)
        root.focus_force()

        # Optional: Escape to exit fullscreen
        root.bind("<Escape>", lambda e: root.attributes("-fullscreen", False))

        # ------------------------ Frames ------------------------
        self.frame = tk.Frame(root, bg="#1E1E2E")
        self.frame.pack(expand=True)

        # Title
        tk.Label(self.frame, text="Sentinel Smart Lock", font=("Helvetica", 24, "bold"),
                 fg="#FFD369", bg="#1E1E2E").pack(pady=(40,10))

        # Subtitle
        tk.Label(self.frame, text="Enter Password to Continue", font=("Arial", 14),
                 fg="#BBBBBB", bg="#1E1E2E").pack(pady=(0,20))

        # Password Entry
        self.password_var = tk.StringVar()
        self.password_entry = tk.Entry(self.frame, textvariable=self.password_var,
                                       font=("Arial", 16), show="*", width=20)
        self.password_entry.pack(pady=10)
        self.password_entry.focus_set()

        # Submit Button
        tk.Button(self.frame, text="SUBMIT", font=("Arial", 14, "bold"), bg="#4ECCA3",
                  fg="white", activebackground="#45B38F", activeforeground="white",
                  width=15, height=2, relief="flat", command=self.check_password).pack(pady=20)

        # Footer
        tk.Label(self.frame, text="Developed by Vadeendra Karanam",
                 font=("Helvetica", 12), fg="#9FB9BE", bg="#1E1E2E").pack(side="bottom", pady=20)

    def check_password(self):
        if self.password_var.get() == "1234":
            self.show_main_buttons()
        else:
            messagebox.showerror("Access Denied", "Incorrect Password!")

    def show_main_buttons(self):
        # Clear current frame
        for widget in self.frame.winfo_children():
            widget.destroy()

        # Title
        tk.Label(self.frame, text="Choose Module to Launch", font=("Helvetica", 20, "bold"),
                 fg="#FFD369", bg="#1E1E2E").pack(pady=(40,20))

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

        tk.Button(self.frame, text="🔑 RFID MODULE",
                  command=lambda: run_module(RFID_PY_PATH, "RFID Module"),
                  **button_style).pack(pady=15)

        tk.Button(self.frame, text="👁 FACE RECOGNITION",
                  command=lambda: run_module(FACE_PY_PATH, "Face Recognition"),
                  **button_style).pack(pady=15)

        tk.Button(self.frame, text="⏹ MAIN SYSTEM",
                  command=lambda: run_module(MAIN_PY_PATH, "Main System"),
                  **button_style, bg="#FF5555", activebackground="#E04747").pack(pady=25)

# ------------------------ START APP ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()
