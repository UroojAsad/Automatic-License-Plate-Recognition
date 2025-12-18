# gui.py
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
from util import read_license_plate

# ------------------ MODEL ------------------
model = YOLO("license_plate_detector.pt")

# ------------------ WINDOW ------------------
root = tk.Tk()
root.title("Smart Parking System | License Plate Recognition")
root.geometry("1050x720")
root.configure(bg="#121212")
root.resizable(False, False)

# ------------------ STYLES ------------------
BG = "#121212"
CARD = "#1E1E1E"
ACCENT = "#00ADB5"
TEXT = "#EEEEEE"
MUTED = "#AAAAAA"
SUCCESS = "#4CAF50"
ERROR = "#F44336"
WARNING = "#FFA726"

FONT_TITLE = ("Segoe UI", 22, "bold")
FONT_SUB = ("Segoe UI", 11)
FONT_BTN = ("Segoe UI", 12, "bold")
FONT_RES = ("Consolas", 14, "bold")

# ------------------ PATHS ------------------
os.makedirs("results", exist_ok=True)
DB_FILE = "results/registered_plates.txt"

# ------------------ STATE ------------------
selected_image_path = ""
current_plate = None
current_crop = None
current_conf = 0.0

# ------------------ DB FUNCTIONS ------------------
def load_registered_plates():
    if not os.path.exists(DB_FILE):
        return set()
    with open(DB_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())

def save_plate_to_db(plate):
    plates = load_registered_plates()
    if plate in plates:
        return False
    with open(DB_FILE, "a") as f:
        f.write(plate + "\n")
    return True

# ------------------ IMAGE SELECT ------------------
def select_image():
    global selected_image_path
    path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )
    if not path:
        return

    selected_image_path = path
    img = Image.open(path).resize((420, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_panel.config(image=img_tk)
    image_panel.image = img_tk

    status_label.config(text="Image Loaded Successfully", fg=SUCCESS)
    result_label.config(text="---", fg=TEXT)
    conf_label.config(text="Confidence: 0.00")
    crop_panel.config(image="")
    register_btn.config(state="disabled")

# ------------------ DETECT ------------------
def detect_plate():
    global current_plate, current_crop, current_conf

    if not selected_image_path:
        status_label.config(text="Please select an image first", fg=ERROR)
        return

    img = cv2.imread(selected_image_path)
    results = model(img)[0]
    detections = results.boxes.data.tolist()

    if not detections:
        status_label.config(text="No License Plate Detected", fg=ERROR)
        return

    x1, y1, x2, y2, score, cls = detections[0]
    h, w = img.shape[:2]
    pad = int(min(h, w) * 0.03)

    crop = img[
        max(0, int(y1 - pad)) : min(h, int(y2 + pad)),
        max(0, int(x1 - pad)) : min(w, int(x2 + pad))
    ]

    text, conf = read_license_plate(crop)

    current_plate = text
    current_crop = crop
    current_conf = conf

    if not text:
        result_label.config(text="NOT FOUND", fg=ERROR)
        conf_label.config(text="Confidence: 0.00")
        status_label.config(text="Plate could not be recognized", fg=ERROR)
        register_btn.config(state="disabled")
        return

    result_label.config(text=text, fg=SUCCESS)
    conf_label.config(text=f"Confidence: {conf:.2f}")

    crop_img = Image.fromarray(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    ).resize((300, 140))
    crop_tk = ImageTk.PhotoImage(crop_img)
    crop_panel.config(image=crop_tk)
    crop_panel.image = crop_tk

    plates_db = load_registered_plates()

    if text in plates_db:
        status_label.config(
            text="Vehicle Allowed ✔ Already Registered",
            fg=SUCCESS
        )
        register_btn.config(state="disabled")
    else:
        status_label.config(
            text="New Vehicle Detected ⚠ Click Register",
            fg=WARNING
        )
        register_btn.config(state="normal")

# ------------------ REGISTER ------------------
def register_vehicle():
    if not current_plate:
        return

    success = save_plate_to_db(current_plate)

    if success:
        status_label.config(
            text="Vehicle Registered Successfully ✅",
            fg=SUCCESS
        )
        register_btn.config(state="disabled")
    else:
        status_label.config(
            text="Duplicate Plate ❌ Registration Blocked",
            fg=ERROR
        )

# ------------------ UI ------------------
header = tk.Label(root, text="SMART PARKING SYSTEM",
                  font=FONT_TITLE, bg=BG, fg=ACCENT)
header.pack(pady=15)

subtitle = tk.Label(
    root,
    text="Automatic License Plate Detection & Recognition",
    font=FONT_SUB,
    bg=BG,
    fg=MUTED
)
subtitle.pack()

main = tk.Frame(root, bg=BG)
main.pack(pady=20)

# LEFT
left = tk.Frame(main, bg=CARD, width=450, height=420)
left.grid(row=0, column=0, padx=20)
left.pack_propagate(False)

image_panel = tk.Label(left, bg=CARD)
image_panel.pack(pady=20)

btn_select = tk.Button(
    left, text="Select Vehicle Image",
    font=FONT_BTN, bg=ACCENT, fg="black",
    relief="flat", width=22,
    command=select_image
)
btn_select.pack(pady=15)

# RIGHT
right = tk.Frame(main, bg=CARD, width=450, height=420)
right.grid(row=0, column=1, padx=20)
right.pack_propagate(False)

plate_title = tk.Label(
    right, text="DETECTED PLATE",
    font=FONT_SUB, bg=CARD, fg=MUTED
)
plate_title.pack(pady=(25, 5))

result_label = tk.Label(
    right, text="---",
    font=FONT_RES, bg=CARD, fg=TEXT
)
result_label.pack()

conf_label = tk.Label(
    right, text="Confidence: 0.00",
    font=FONT_SUB, bg=CARD, fg=MUTED
)
conf_label.pack(pady=5)

crop_panel = tk.Label(right, bg=CARD)
crop_panel.pack(pady=15)

btn_detect = tk.Button(
    right, text="Detect License Plate",
    font=FONT_BTN, bg=SUCCESS, fg="black",
    relief="flat", width=22,
    command=detect_plate
)
btn_detect.pack(pady=10)

register_btn = tk.Button(
    right, text="Register Vehicle",
    font=FONT_BTN, bg=WARNING, fg="black",
    relief="flat", width=22,
    state="disabled",
    command=register_vehicle
)
register_btn.pack(pady=10)

status_label = tk.Label(
    root, text="Waiting for input...",
    font=FONT_SUB, bg=BG, fg=MUTED
)
status_label.pack(pady=15)

root.mainloop()
