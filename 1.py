#!/usr/bin/env python3
import tkinter as tk
from tkinter import messagebox
import subprocess
import os

# ------------------------ PATHS ------------------------
MAIN_PY_PATH = "/home/project/Desktop/Att/1.py"
RFID_PY_PATH = "/home/project/Desktop/Att/rfid.py"
FACE_PY_PATH = "/home/project/Desktop/Att/face.py"

# ------------------------ GUI ------------------------
class LauncherGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔒 Sentinel Smart Lock Launcher")
        self.root.attributes("-fullscreen", True)
        self.root.configure(bg="#1E1E2E")

        self.correct_password = "1234"

        self.build_password_screen()

    # ---------------- PASSWORD SCREEN ----------------
    def build_password_screen(self):
        self.clear_root()

        tk.Label(
            self.root,
            text="Sentinel Smart Lock",
            font=("Helvetica", 24, "bold"),
            fg="#FFD369",
            bg="#1E1E2E",
        ).pack(pady=40)

        tk.Label(
            self.root,
            text="Enter Password",
            font=("Arial", 16),
            fg="#BBBBBB",
            bg="#1E1E2E",
        ).pack(pady=10)

        self.password_entry = tk.Entry(self.root, show="*", font=("Arial", 16), width=15)
        self.password_entry.pack(pady=10)
        self.password_entry.focus()

        submit_btn = tk.Button(
            self.root,
            text="Submit",
            font=("Arial", 14, "bold"),
            bg="#4ECCA3",
            fg="white",
            activebackground="#45B38F",
            width=15,
            command=self.check_password
        )
        submit_btn.pack(pady=20)

    # ---------------- CHECK PASSWORD ----------------
    def check_password(self):
        if self.password_entry.get() == self.correct_password:
            self.build_main_screen()
        else:
            messagebox.showerror("Access Denied", "Incorrect Password!")
            self.password_entry.delete(0, tk.END)

    # ---------------- MAIN SCREEN ----------------
    def build_main_screen(self):
        self.clear_root()

        tk.Label(
            self.root,
            text="Sentinel Smart Lock",
            font=("Helvetica", 24, "bold"),
            fg="#FFD369",
            bg="#1E1E2E",
        ).pack(pady=20)

        subtitle = tk.Label(
            self.root,
            text="Choose Module to Launch",
            font=("Arial", 14),
            fg="#BBBBBB",
            bg="#1E1E2E",
        )
        subtitle.pack(pady=10)

        # Button style
        btn_style = {"font": ("Arial", 13, "bold"), "bg": "#4ECCA3", "fg": "white",
                     "activebackground": "#45B38F", "width": 20, "height": 2}

        tk.Button(self.root, text="🔑 RFID MODULE", command=self.run_rfid, **btn_style).pack(pady=10)
        tk.Button(self.root, text="👁 FACE RECOGNITION", command=self.run_face, **btn_style).pack(pady=10)

        tk.Button(self.root, text="⏹ CLOSE & RETURN", command=self.close_launcher,
                  font=("Arial", 13, "bold"), bg="#FF5555", fg="white",
                  activebackground="#E04747", width=20, height=2).pack(pady=15)

    # ---------------- UTILITY ----------------
    def clear_root(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def run_rfid(self):
        subprocess.Popen(["python3", RFID_PY_PATH])
        messagebox.showinfo("RFID Module", "RFID system launched successfully!")

    def run_face(self):
        subprocess.Popen(["python3", FACE_PY_PATH])
        messagebox.showinfo("Face Module", "Face Recognition system launched successfully!")

    def close_launcher(self):
        self.root.destroy()
        subprocess.Popen(["python3", MAIN_PY_PATH])

# ------------------------ RUN ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherGUI(root)
    root.mainloop()
