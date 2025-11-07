#!/usr/bin/env python3
# Fullscreen Smart Lock Launcher with Password + Auto Return

import tkinter as tk
from tkinter import messagebox
import subprocess

# ------------------------ PATHS ------------------------
MAIN_PY_PATH = "/home/project/Desktop/Att/1.py"
RFID_PY_PATH = "/home/project/Desktop/Att/rfid.py"
FACE_PY_PATH = "/home/project/Desktop/Att/face.py"

AUTO_RETURN_TIME = 120  # 2 minutes

# ------------------------ MODULE LAUNCHERS ------------------------
def run_module(path, name):
    """Run a module and return to main system after it exits"""
    try:
        proc = subprocess.Popen(["python3", path])
        proc.wait()  # Wait until module exits
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch {name}:\n{e}")

# ------------------------ GUI ------------------------
class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔒 Sentinel Smart Lock Launcher")
        self.root.configure(bg="#1E1E2E")

        # Start in fullscreen
        self.fullscreen = True
        self.root.attributes("-fullscreen", True)
        self.root.focus_force()
        self.root.bind("<Escape>", self.toggle_fullscreen)

        self.frame = tk.Frame(root, bg="#1E1E2E")
        self.frame.pack(expand=True, fill="both")

        self.password_screen()

        # Start auto-return timer
        self.auto_return_id = self.root.after(AUTO_RETURN_TIME * 1000, self.return_home)

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def password_screen(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

        tk.Label(self.frame, text="Sentinel Smart Lock", font=("Helvetica", 24, "bold"),
                 fg="#FFD369", bg="#1E1E2E").pack(pady=(40,10))
        tk.Label(self.frame, text="Enter Password to Continue", font=("Arial", 14),
                 fg="#BBBBBB", bg="#1E1E2E").pack(pady=(0,20))

        self.password_var = tk.StringVar()
        self.password_entry = tk.Entry(self.frame, textvariable=self.password_var,
                                       font=("Arial", 16), show="*", width=20)
        self.password_entry.pack(pady=10)
        self.password_entry.focus_set()

        tk.Button(self.frame, text="SUBMIT", font=("Arial", 14, "bold"), bg="#4ECCA3",
                  fg="white", activebackground="#45B38F", activeforeground="white",
                  width=15, height=2, relief="flat", command=self.check_password).pack(pady=20)

        tk.Label(self.frame, text="Developed by Vadeendra Karanam",
                 font=("Helvetica", 12), fg="#9FB9BE", bg="#1E1E2E").pack(side="bottom", pady=20)

    def check_password(self):
        if self.password_var.get() == "1234":
            self.show_main_buttons()
        else:
            messagebox.showerror("Access Denied", "Incorrect Password!")

    def show_main_buttons(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

        tk.Label(self.frame, text="Choose Module to Launch", font=("Helvetica", 20, "bold"),
                 fg="#FFD369", bg="#1E1E2E").pack(pady=(40,20))

        # Base button style
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

        # Buttons
        tk.Button(self.frame, text="🔑 RFID MODULE",
                  command=lambda: run_module(RFID_PY_PATH, "RFID Module"),
                  bg="#4ECCA3", activebackground="#45B38F", **button_style).pack(pady=10)

        tk.Button(self.frame, text="👁 FACE RECOGNITION",
                  command=lambda: run_module(FACE_PY_PATH, "Face Recognition"),
                  bg="#4ECCA3", activebackground="#45B38F", **button_style).pack(pady=10)

        tk.Button(self.frame, text="⏹ MAIN SYSTEM",
                  command=lambda: run_module(MAIN_PY_PATH, "Main System"),
                  bg="#FF5555", activebackground="#E04747", **button_style).pack(pady=10)

        tk.Button(self.frame, text="🏠 RETURN TO HOME",
                  command=self.return_home,
                  bg="#FFA500", activebackground="#E59400", **button_style).pack(pady=10)

        tk.Button(self.frame, text="❌ TERMINATE & GO TO 1.py",
                  command=self.terminate_and_go_main,
                  bg="#FF4444", activebackground="#CC0000", **button_style).pack(pady=10)

        # Reset auto-return timer whenever user interacts
        self.frame.bind_all("<Button-1>", self.reset_timer)
        self.frame.bind_all("<Key>", self.reset_timer)

    def return_home(self, event=None):
        """Return to main smart lock GUI"""
        self.root.destroy()
        subprocess.Popen(["python3", MAIN_PY_PATH])

    def terminate_and_go_main(self):
        """Terminate this launcher and open 1.py"""
        self.root.destroy()
        subprocess.Popen(["python3", MAIN_PY_PATH])

    def reset_timer(self, event=None):
        """Reset auto-return timer on user interaction"""
        if self.auto_return_id:
            self.root.after_cancel(self.auto_return_id)
        self.auto_return_id = self.root.after(AUTO_RETURN_TIME * 1000, self.return_home)


# ------------------------ START APP ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()
