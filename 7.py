#!/usr/bin/env python3
# Smart Lock Launcher with Password Login

import tkinter as tk
from tkinter import messagebox
import subprocess

# ------------------------ PATHS ------------------------
MAIN_PY_PATH = "/home/project/Desktop/Att/1.py"
RFID_PY_PATH = "/home/project/Desktop/Att/rfid.py"
FACE_PY_PATH = "/home/project/Desktop/Att/face.py"

AUTO_RETURN_TIME = 120  # 2 minutes
PASSWORD = "1234"       # ---- SET YOUR PASSWORD HERE ----


# ------------------------ MODULE LAUNCHER ------------------------
def run_module(path, name):
    try:
        subprocess.Popen(["python3", path])
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch {name}:\n{e}")


# ------------------------ LOGIN WINDOW ------------------------
class LoginWindow:
    def __init__(self, root):
        self.root = root
        self.root.title("Login")
        self.root.configure(bg="#1E1E2E")

        self.root.attributes("-fullscreen", True)

        frame = tk.Frame(root, bg="#1E1E2E")
        frame.pack(expand=True)

        tk.Label(frame, text="Enter Password",
                 font=("Arial", 22, "bold"), fg="#FFD369", bg="#1E1E2E").pack(pady=20)

        self.pwd_entry = tk.Entry(frame, show="*", font=("Arial", 18), width=20)
        self.pwd_entry.pack(pady=10)
        self.pwd_entry.focus()

        tk.Button(frame, text="Login", font=("Arial", 16, "bold"),
                  bg="#4ECCA3", fg="white",
                  command=self.check_password,
                  width=12, height=1).pack(pady=30)

        # Allow Enter key to submit
        self.root.bind("<Return>", self.check_password)

    def check_password(self, event=None):
        if self.pwd_entry.get() == PASSWORD:
            self.root.destroy()

            # Open launcher
            launcher_root = tk.Tk()
            app = LauncherApp(launcher_root)
            launcher_root.mainloop()
        else:
            messagebox.showerror("Error", "Incorrect Password!")
            self.pwd_entry.delete(0, tk.END)


# ------------------------ LAUNCHER APP ------------------------
class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔒 Sentinel Smart Lock Launcher")
        self.root.configure(bg="#1E1E2E")

        self.root.attributes("-fullscreen", True)
        self.root.focus_force()
        self.root.bind("<Escape>", lambda e: self.root.attributes("-fullscreen", False))

        self.frame = tk.Frame(root, bg="#1E1E2E")
        self.frame.pack(expand=True, fill="both")

        self.show_main_buttons()

        self.auto_return_id = self.root.after(AUTO_RETURN_TIME * 1000, self.return_home)

    def show_main_buttons(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

        tk.Label(self.frame, text="Sentinel Smart Lock Launcher",
                 font=("Helvetica", 24, "bold"), fg="#FFD369", bg="#1E1E2E").pack(pady=(40,20))

        button_style = {
            "font": ("Arial", 16, "bold"),
            "fg": "white",
            "activeforeground": "white",
            "relief": "flat",
            "width": 25,
            "height": 2,
            "bd": 0,
            "cursor": "hand2",
        }

        tk.Button(self.frame, text="🔑 RFID MODULE",
                  command=lambda: run_module(RFID_PY_PATH, "RFID Module"),
                  bg="#4ECCA3", activebackground="#45B38F", **button_style).pack(pady=20)

        tk.Button(self.frame, text="👁 FACE RECOGNITION",
                  command=lambda: run_module(FACE_PY_PATH, "Face Recognition"),
                  bg="#4ECCA3", activebackground="#45B38F", **button_style).pack(pady=20)

        tk.Button(self.frame, text="Go to Home",
                  command=self.terminate_and_go_main,
                  bg="#FF4444", activebackground="#CC0000", **button_style).pack(pady=20)

        self.frame.bind_all("<Button-1>", self.reset_timer)
        self.frame.bind_all("<Key>", self.reset_timer)

    def return_home(self, event=None):
        self.root.destroy()
        subprocess.Popen(["python3", MAIN_PY_PATH])

    def terminate_and_go_main(self):
        if self.auto_return_id:
            self.root.after_cancel(self.auto_return_id)
        self.root.destroy()
        subprocess.Popen(["python3", MAIN_PY_PATH])

    def reset_timer(self, event=None):
        if self.auto_return_id:
            self.root.after_cancel(self.auto_return_id)
        self.auto_return_id = self.root.after(AUTO_RETURN_TIME * 1000, self.return_home)


# ------------------------ START PROGRAM ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    LoginWindow(root)
    root.mainloop()
