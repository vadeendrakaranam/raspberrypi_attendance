import sys
import os
import tkinter as tk
from tkinter import messagebox
import csv
import logging
import numpy as np
import cv2
import dlib
from filelock import FileLock

# ---------------------- Paths ----------------------
LIB_DIR = os.path.abspath("/home/project/Desktop/Att/lib")
DATA_DIR = os.path.join("/home/project/Desktop/Att", "data")
PATH_IMAGES_FROM_CAMERA = os.path.join(DATA_DIR, "data_faces_from_camera")
CSV_PATH = os.path.join(DATA_DIR, "features_all.csv")
LOCK_PATH = os.path.join(LIB_DIR, "features_all.csv.lock")

os.makedirs(PATH_IMAGES_FROM_CAMERA, exist_ok=True)
os.makedirs(LIB_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# ---------------------- Dlib setup ----------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join(DATA_DIR, "data_dlib/shape_predictor_68_face_landmarks.dat"))
face_reco_model = dlib.face_recognition_model_v1(os.path.join(DATA_DIR, "data_dlib/dlib_face_recognition_resnet_model_v1.dat"))

# ---------------------- Functions ----------------------
def return_128d_features(img_path):
    img_rd = cv2.imread(img_path)
    faces = detector(img_rd, 1)
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
        return face_descriptor
    else:
        logging.warning("No face detected in %s", img_path)
        return None

def return_features_mean_personX(person_folder):
    features_list = []
    photos = os.listdir(person_folder)
    for photo in photos:
        img_path = os.path.join(person_folder, photo)
        logging.info("Reading image: %s", img_path)
        features = return_128d_features(img_path)
        if features is not None:
            features_list.append(features)
    if features_list:
        return np.array(features_list).mean(axis=0)
    else:
        return np.zeros(128)

def save_features_to_csv(person_name, features):
    with FileLock(LOCK_PATH, timeout=10):
        file_exists = os.path.exists(CSV_PATH)
        with open(CSV_PATH, "a", newline="") as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists:
                header = ["name"] + [f"f{i}" for i in range(128)]
                writer.writerow(header)
            writer.writerow([person_name] + features.tolist())
        logging.info("✅ Saved %s safely to %s", person_name, CSV_PATH)

def extract_features_for_selected(folders):
    for folder in folders:
        folder_path = os.path.join(PATH_IMAGES_FROM_CAMERA, folder)
        if os.path.exists(folder_path):
            features_mean = return_features_mean_personX(folder_path)
            save_features_to_csv(folder, features_mean)
            logging.info(f"✅ Features for {folder} extracted and saved.")
        else:
            logging.warning(f"❌ Folder not found: {folder}")
    messagebox.showinfo("Done", "✅ Feature extraction completed for selected folders.")

# ---------------------- GUI ----------------------
def main_gui():
    root = tk.Tk()
    root.title("Select Person Folders")
    root.geometry("400x400")

    tk.Label(root, text="Select Folder(s) to Extract Features:").pack(pady=10)

    listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=50, height=15)
    listbox.pack(pady=10, padx=20, fill="both")

    # Populate listbox with folders sorted newest first
    folders = [f for f in os.listdir(PATH_IMAGES_FROM_CAMERA) if os.path.isdir(os.path.join(PATH_IMAGES_FROM_CAMERA, f))]
    folders.sort(key=lambda f: os.path.getctime(os.path.join(PATH_IMAGES_FROM_CAMERA, f)), reverse=True)
    for folder in folders:
        listbox.insert(tk.END, folder)

    def on_extract():
        selected_indices = listbox.curselection()
        if not selected_indices:
            messagebox.showwarning("No Selection", "Please select at least one folder.")
            return
        selected_folders = [listbox.get(i) for i in selected_indices]
        root.withdraw()  # hide main window
        extract_features_for_selected(selected_folders)
        root.destroy()

    tk.Button(root, text="Extract Features", command=on_extract).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main_gui()
