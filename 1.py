#!/usr/bin/env python3
# RFID Tag Manager GUI

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
root = tk.Tk()
root.title("RFID Tag Manager")
root.attributes("-fullscreen", True)  # Fullscreen
root.configure(bg="#1E1E2E")

# Press ESC to exit fullscreen
def toggle_fullscreen(event=None):
    root.attributes("-fullscreen", False)
root.bind("<Escape>", toggle_fullscreen)

# ---------------- Tag Operations ----------------
def add_tag():
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

def update_tag():
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

def show_all_tags():
    c.execute("SELECT tag_id, name FROM rfid_users")
    rows = c.fetchall()
    if not rows:
        messagebox.showinfo("All Tags", "No tags stored.")
        return
    info = "\n".join([f"{name} ({tag_id})" for tag_id, name in rows])
    messagebox.showinfo("All Tags", info)

def delete_tag_popup():
    popup = tk.Toplevel(root)
    popup.title("Delete Tags")
    popup.geometry("400x350")
    
    c.execute("SELECT tag_id, name FROM rfid_users")
    rows = c.fetchall()
    
    if not rows:
        tk.Label(popup, text="No tags stored.").pack(pady=20)
        return
    
    for tag_id, name in rows:
        frame = tk.Frame(popup)
        frame.pack(fill="x", pady=2, padx=5)
        tk.Label(frame, text=f"{name} ({tag_id})", anchor="w").pack(side="left")
        tk.Button(frame, text="Delete", command=lambda t=tag_id, n=name: delete_tag_confirm(t, n, popup)).pack(side="right", padx=5)

def delete_tag_confirm(tag_id, name, popup):
    confirm = messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete tag '{name}'?")
    if confirm:
        c.execute("DELETE FROM rfid_users WHERE tag_id=?", (tag_id,))
        conn.commit()
        messagebox.showinfo("Deleted", f"Tag '{name}' deleted successfully")
        popup.destroy()
        delete_tag_popup()  # refresh

def close_app():
    conn.close()
    root.destroy()
    # Automatically open main lock script
    os.system("python3 /home/project/Desktop/Att/1.py &")

# ---------------- Button Style ----------------
button_style = {
    "font": ("Arial", 14, "bold"),
    "bg": "#4ECCA3",
    "fg": "white",
    "activebackground": "#45B38F",
    "activeforeground": "white",
    "relief": "flat",
    "width": 25,
    "height": 2,
    "bd": 0,
    "cursor": "hand2",
}

# ---------------- Buttons ----------------
tk.Button(root, text="Add Tag", command=add_tag, **button_style).pack(pady=8)
tk.Button(root, text="Update Tag", command=update_tag, **button_style).pack(pady=8)
tk.Button(root, text="Delete Tag", command=delete_tag_popup, **button_style).pack(pady=8)
tk.Button(root, text="All Tags", command=show_all_tags, **button_style).pack(pady=8)

# Special red close button
close_btn_style = button_style.copy()
close_btn_style["bg"] = "#FF5555"
close_btn_style["activebackground"] = "#E04747"
tk.Button(root, text="Close & Open Main Lock", command=close_app, **close_btn_style).pack(pady=20)

# ---------------- Run GUI ----------------
root.mainloop()
