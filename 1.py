import sys
import os
import shutil
import time
import logging
import tkinter as tk
from tkinter import font as tkFont, messagebox
from PIL import Image, ImageTk
import dlib
import numpy as np
import cv2

# Add your lib path if needed
sys.path.insert(0, os.path.abspath("/home/project/Desktop/Att/lib"))

# ---------------- Face Detector ----------------
detector = dlib.get_frontal_face_detector()

# ---------------- Camera Auto-detect ----------------
def find_camera_index(max_index=5):
    """Return first available camera index, else None"""
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release()
            return i
    return None

CAMERA_INDEX = find_camera_index()
if CAMERA_INDEX is None:
    print("No camera detected!")
    tk.Tk().withdraw()
    messagebox.showerror("Camera Error", "No camera detected! Exiting.")
    sys.exit(1)
else:
    print(f"Camera detected at index {CAMERA_INDEX}")

# ---------------- Face Register Class ----------------
class Face_Register:
    def __init__(self):
        self.current_frame_faces_cnt = 0
        self.existing_faces_cnt = 0
        self.ss_cnt = 0
        self.face_folder_created_flag = False
        self.out_of_range_flag = False

        # Tkinter GUI
        self.win = tk.Tk()
        self.win.title("Face Register")
        self.win.geometry("1000x500")

        # Left frame for camera feed
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        # Right frame for info and controls
        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_name_char = ""
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        # Current frame and face ROI
        self.current_frame = np.ndarray
        self.face_ROI_image = np.ndarray
        self.face_ROI_width_start = 0
        self.face_ROI_height_start = 0
        self.face_ROI_width = 0
        self.face_ROI_height = 0
        self.ww = 0
        self.hh = 0

        # FPS calculation
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # ---------------- Video Capture ----------------
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera at index {CAMERA_INDEX}")
            self.win.destroy()
            sys.exit(1)

    # ---------------- Folder Setup ----------------
    def pre_work_mkdir(self):
        if not os.path.isdir(self.path_photos_from_camera):
            os.makedirs(self.path_photos_from_camera)

    def check_existing_faces_cnt(self):
        folder_path = self.path_photos_from_camera
        if os.path.isdir(folder_path) and os.listdir(folder_path):
            person_list = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            person_num_list = []
            for person in person_list:
                parts = person.split('_')
                if len(parts) >= 2:
                    try:
                        person_order = int(parts[1])
                        person_num_list.append(person_order)
                    except ValueError:
                        pass
            self.existing_faces_cnt = max(person_num_list) if person_num_list else 0
        else:
            self.existing_faces_cnt = 0

    # ---------------- GUI Methods ----------------
    def GUI_clear_data(self):
        folders = os.listdir(self.path_photos_from_camera)
        for f in folders:
            shutil.rmtree(os.path.join(self.path_photos_from_camera, f))
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.existing_faces_cnt = 0
        self.label_cnt_face_in_database['text'] = "0"
        self.log_all["text"] = "Face images and `features_all.csv` removed!"

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_info(self):
        tk.Label(self.frame_right_info, text="Face register", font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)
        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Label(self.frame_right_info, text="Faces in database: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)
        tk.Label(self.frame_right_info, text="Faces in current frame: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)
        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step buttons
        tk.Label(self.frame_right_info, font=self.font_step_title, text="Step 1: Clear face photos").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Button(self.frame_right_info, text='Clear', command=self.GUI_clear_data).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, font=self.font_step_title, text="Step 2: Input name").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Label(self.frame_right_info, text="Name: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)
        tk.Button(self.frame_right_info, text='Input', command=self.GUI_get_input_name).grid(row=8, column=2, padx=5)

        tk.Label(self.frame_right_info, font=self.font_step_title, text="Step 3: Save face image").grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Button(self.frame_right_info, text='Save current face', command=self.save_current_face).grid(row=10, column=0, columnspan=3, sticky=tk.W)
        self.log_all.grid(row=11, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)
        self.frame_right_info.pack()

    def create_face_folder(self):
        self.existing_faces_cnt += 1
        if self.input_name_char:
            self.current_face_dir = os.path.join(self.path_photos_from_camera, f"person_{self.existing_faces_cnt}_{self.input_name_char}")
        else:
            self.current_face_dir = os.path.join(self.path_photos_from_camera, f"person_{self.existing_faces_cnt}")
        os.makedirs(self.current_face_dir, exist_ok=True)
        self.log_all["text"] = f"\"{self.current_face_dir}/\" created!"
        self.ss_cnt = 0
        self.face_folder_created_flag = True

    # ---------------- Face Saving ----------------
    def save_current_face(self):
        if self.face_folder_created_flag and self.current_frame_faces_cnt == 1 and not self.out_of_range_flag:
            self.ss_cnt += 1
            face_img = self.current_frame[self.face_ROI_height_start - self.hh : self.face_ROI_height_start - self.hh + self.face_ROI_height*2,
                                          self.face_ROI_width_start - self.ww : self.face_ROI_width_start - self.ww + self.face_ROI_width*2]
            self.face_ROI_image = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(self.current_face_dir, f"img_face_{self.ss_cnt}.jpg"), self.face_ROI_image)
            self.log_all["text"] = f"Saved img_face_{self.ss_cnt}.jpg"
        elif not self.face_folder_created_flag:
            self.log_all["text"] = "Please run Step 2 first!"
        elif self.current_frame_faces_cnt != 1:
            self.log_all["text"] = "No face or multiple faces detected!"
        elif self.out_of_range_flag:
            self.log_all["text"] = "Face out of range!"

    # ---------------- Video Capture ----------------
    def get_frame(self):
        if self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (480, 360))
                return ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return False, None

    # ---------------- Process Loop ----------------
    def process(self):
        ret, self.current_frame = self.get_frame()
        if not ret or self.current_frame is None:
            self.win.after(20, self.process)
            return

        faces = detector(self.current_frame, 0)
        self.update_fps()
        self.label_face_cnt["text"] = str(len(faces))

        for d in faces:
            self.face_ROI_width_start = d.left()
            self.face_ROI_height_start = d.top()
            self.face_ROI_height = d.bottom() - d.top()
            self.face_ROI_width = d.right() - d.left()
            self.hh = self.face_ROI_height // 2
            self.ww = self.face_ROI_width // 2

            # Out of range check
            if (d.right() + self.ww > 640 or d.bottom() + self.hh > 480 or d.left() - self.ww < 0 or d.top() - self.hh < 0):
                self.label_warning["text"] = "OUT OF RANGE"
                self.label_warning['fg'] = 'red'
                self.out_of_range_flag = True
            else:
                self.label_warning["text"] = ""
                self.out_of_range_flag = False
                self.current_frame = cv2.rectangle(self.current_frame,
                                                   (d.left() - self.ww, d.top() - self.hh),
                                                   (d.right() + self.ww, d.bottom() + self.hh),
                                                   (255, 255, 255), 2)

        self.current_frame_faces_cnt = len(faces)
        img = ImageTk.PhotoImage(Image.fromarray(self.current_frame))
        self.label.img_tk = img
        self.label.configure(image=img)
        self.win.after(20, self.process)

    # ---------------- FPS ----------------
    def update_fps(self):
        now = time.time()
        if int(now) != int(self.start_time):
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        if self.frame_time != 0:
            self.fps = 1.0 / self.frame_time
        self.frame_start_time = now
        self.label_fps_info["text"] = str(round(self.fps, 2))

    # ---------------- Run ----------------
    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()
        self.win.mainloop()


# ---------------- Main ----------------
def main():
    logging.basicConfig(level=logging.INFO)
    Face_Register_con = Face_Register()
    Face_Register_con.run()


if __name__ == '__main__':
    main()
