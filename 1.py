#!/usr/bin/env python3
# Fullscreen Smart Lock Launcher with Password + Auto Return

import tkinter as tk
from tkinter import messagebox
import subprocess

# ------------------------ PATHS ------------------------
MAIN_PY_PATH = "/home/project/Desktop/Att/1.py"
RFID_PY_PATH = "/home/project/Desktop/Att/rfid.py"
FACE_PY_PATH = "/home/project/Desktop/Att/face.py"

# ------------------------ MODULE LAUNCHERS ------------------------
def run_module_and_return(path, name, launcher_root):
    """Run a module and return to main system after it exits"""
    try:
        proc = subprocess.Popen(["python3", path])
        launcher_root.withdraw()  # Hide launcher while module runs
        proc.wait()  # Wait until module exits
        launcher_root.destroy()  # Close launcher
        subprocess.Popen(["python3", MAIN_PY_PATH])  # Start main system
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
        # Toggle fullscreen with ESC
        self.root.bind("<Escape>", self.toggle_fullscreen)

        # Frame container
        self.frame = tk.Frame(root, bg="#1E1E2E")
        self.frame.pack(expand=True, fill="both")

        # Password screen
        self.password_screen()

    def toggle_fullscreen(self, event=None):
        """Toggle fullscreen on ESC key"""
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def password_screen(self):
        """Display password entry screen"""
        # Clear frame
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
        """Display module selection buttons"""
        # Clear frame
        for widget in self.frame.winfo_children():
            widget.destroy()

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
                  command=lambda: run_module_and_return(RFID_PY_PATH, "RFID Module", self.root),
                  **button_style).pack(pady=15)

        tk.Button(self.frame, text="👁 FACE RECOGNITION",
                  command=lambda: run_module_and_return(FACE_PY_PATH, "Face Recognition", self.root),
                  **button_style).pack(pady=15)

        tk.Button(self.frame, text="⏹ MAIN SYSTEM",
                  command=lambda: run_module_and_return(MAIN_PY_PATH, "Main System", self.root),
                  **button_style, bg="#FF5555", activebackground="#E04747").pack(pady=25)

# ------------------------ START APP ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop()
