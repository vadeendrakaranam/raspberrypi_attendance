#!/usr/bin/env python3
# Fullscreen RFID Tag Manager

import tkinter as tk
from tkinter import simpledialog, messagebox
import sqlite3
from mfrc522 import SimpleMFRC522
import os

# ---------------- Database Setup ----------------
DB_PATH = os.path.expanduser("~/Desktop/Att/rfid_data.db")
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS rfid_users (
        tag_id TEXT PRIMARY KEY,
        name TEXT
    )
""")
conn.commit()

# ---------------- RFID Reader ----------------
reader = SimpleMFRC522()

# ---------------- Tkinter GUI ----------------
class RFIDManager:
    def __init__(self, root):
        self.root = root
        self.root.title("RFID Tag Manager")
        self.root.configure(bg="#1E1E2E")
        
        # Fullscreen setup
        self.fullscreen = True
        self.root.attributes("-fullscreen", True)
        self.root.focus_force()
        self.root.bind("<Escape>", self.toggle_fullscreen)

        # Frame container
        self.frame = tk.Frame(root, bg="#1E1E2E")
        self.frame.pack(expand=True, fill="both")

        self.setup_ui()

    def toggle_fullscreen(self, event=None):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def setup_ui(self):
        # Title
        tk.Label(self.frame, text="RFID Tag Manager", font=("Helvetica", 24, "bold"),
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

        tk.Button(self.frame, text="Add Tag", command=self.add_tag, **button_style).pack(pady=10)
        tk.Button(self.frame, text="Update Tag", command=self.update_tag, **button_style).pack(pady=10)
        tk.Button(self.frame, text="Delete Tag", command=self.delete_tag_popup, **button_style).pack(pady=10)
        tk.Button(self.frame, text="Show All Tags", command=self.show_all_tags, **button_style).pack(pady=10)
        tk.Button(self.frame, text="Close", command=self.close_app, **button_style, bg="#FF5555", activebackground="#E04747").pack(pady=20)

    # ---------------- Tag Operations ----------------
    def add_tag(self):
        messagebox.showinfo("Info", "Scan new RFID tag...")
        try:
            uid, _ = reader.read()
            uid = str(uid)
            c.execute("SELECT name FROM rfid_users WHERE tag_id=?", (uid,))
            row = c.fetchone()
            if row:
                messagebox.showinfo("Info", f"Tag already exists:\nUID: {uid}\nName: {row[0]}")
            else:
                name = simpledialog.askstring("Add Tag", "Enter name for this tag:")
                if name:
                    c.execute("INSERT INTO rfid_users(tag_id, name) VALUES(?, ?)", (uid, name))
                    conn.commit()
                    messagebox.showinfo("Success", f"Tag '{name}' added with UID {uid}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def update_tag(self):
        messagebox.showinfo("Info", "Scan RFID tag to update...")
        try:
            uid, _ = reader.read()
            uid = str(uid)
            c.execute("SELECT name FROM rfid_users WHERE tag_id=?", (uid,))
            row = c.fetchone()
            if row:
                new_name = simpledialog.askstring("Update Tag", f"Current name: {row[0]}\nEnter new name:")
                if new_name:
                    c.execute("UPDATE rfid_users SET name=? WHERE tag_id=?", (new_name, uid))
                    conn.commit()
                    messagebox.showinfo("Success", f"Tag updated to '{new_name}'")
            else:
                messagebox.showwarning("Unknown Card", f"Tag UID {uid} not found!")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_all_tags(self):
        c.execute("SELECT tag_id, name FROM rfid_users")
        rows = c.fetchall()
        if not rows:
            messagebox.showinfo("All Tags", "No tags stored.")
            return
        info = "\n".join([f"{name} ({tag_id})" for tag_id, name in rows])
        messagebox.showinfo("All Tags", info)

    def delete_tag_popup(self):
        popup = tk.Toplevel(self.root)
        popup.title("Delete Tags")
        popup.geometry("400x400")
        popup.configure(bg="#1E1E2E")

        c.execute("SELECT tag_id, name FROM rfid_users")
        rows = c.fetchall()

        if not rows:
            tk.Label(popup, text="No tags stored.", fg="white", bg="#1E1E2E", font=("Arial",14)).pack(pady=20)
            return

        for tag_id, name in rows:
            frame = tk.Frame(popup, bg="#1E1E2E")
            frame.pack(fill="x", pady=2, padx=5)
            tk.Label(frame, text=f"{name} ({tag_id})", anchor="w", fg="white", bg="#1E1E2E", font=("Arial",12)).pack(side="left")
            tk.Button(frame, text="Delete", command=lambda t=tag_id, n=name: self.delete_tag_confirm(t, n, popup),
                      bg="#FF5555", fg="white", activebackground="#E04747", width=10).pack(side="right", padx=5)

    def delete_tag_confirm(self, tag_id, name, popup):
        confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete tag '{name}'?")
        if confirm:
            c.execute("DELETE FROM rfid_users WHERE tag_id=?", (tag_id,))
            conn.commit()
            messagebox.showinfo("Deleted", f"Tag '{name}' deleted successfully")
            popup.destroy()
            self.delete_tag_popup()  # refresh pop-up

    def close_app(self):
        conn.close()
        self.root.destroy()


# ------------------------ START ------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = RFIDManager(root)
    root.mainloop()
