# gui.py
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import os
import hashlib
from util import read_license_plate

# Load model (ensure license_plate_detector.pt exists in same folder)
model = YOLO("license_plate_detector.pt")

# GUI window
root = tk.Tk()
root.title("License Plate Detection & OCR (GUI)")
root.geometry("940x700")
root.configure(bg="#1e1e1e")

selected_image_path = ""
# Sets to avoid duplicate saves (optional persisted file could be added later)
saved_hashes = set()
saved_texts = set()

# Create result dirs
os.makedirs("results/all_plates", exist_ok=True)
os.makedirs("results/text", exist_ok=True)
csv_log_path = "results/detection_log.csv"

# Ensure CSV header exists
if not os.path.exists(csv_log_path):
    with open(csv_log_path, "w", encoding="utf-8") as f:
        f.write("Image,PlateText,Confidence,BBox,SavedImage\n")


def calculate_hash(image: np.ndarray) -> str:
    """SHA-256 hash for numpy image bytes"""
    return hashlib.sha256(image.tobytes()).hexdigest()


def save_plate_and_text(cropped_img: np.ndarray, text: str, conf: float, src_image_path: str, bbox) -> tuple:
    """Saves unique plate images and text. Returns tuple(saved_image_path, saved_text_flag)"""
    saved_image_path = ""
    saved_text_flag = False

    # image hash
    img_hash = calculate_hash(cropped_img)
    if img_hash not in saved_hashes:
        saved_image_path = f"results/all_plates/{img_hash[:12]}.jpg"
        cv2.imwrite(saved_image_path, cropped_img)
        saved_hashes.add(img_hash)

    # text
    text = (text or "").strip()
    if text and (text not in saved_texts):
        saved_texts.add(text)
        with open("results/text/plates_text.txt", "a", encoding="utf-8") as f:
            f.write(f"{text},{conf}\n")
        saved_text_flag = True

    # append to CSV log
    with open(csv_log_path, "a", encoding="utf-8") as f:
        f.write(f"\"{src_image_path}\",\"{text}\",{conf},\"{bbox}\",\"{saved_image_path}\"\n")

    return saved_image_path, saved_text_flag


def select_image():
    global selected_image_path
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp")])
    if not path:
        return
    selected_image_path = path
    pil = Image.open(path).resize((420, 320))
    img_tk = ImageTk.PhotoImage(pil)
    panel.configure(image=img_tk)
    panel.image = img_tk
    result_label.config(text="Image selected: " + os.path.basename(path), fg="white")


def detect_plate():
    if selected_image_path == "":
        result_label.config(text="⚠️ Please select an image first!", fg="red")
        return

    img = cv2.imread(selected_image_path)
    results = model(img)[0]
    plates = results.boxes.data.tolist()
    if not plates:
        result_label.config(text="❌ No plate detected!", fg="red")
        return

    # take top-scoring detection
    x1, y1, x2, y2, score, cls = plates[0]
    # add small padding to avoid cropping out characters
    h, w = img.shape[:2]
    pad = max(6, int(min(h, w) * 0.03))
    x1p = max(0, int(x1) - pad)
    y1p = max(0, int(y1) - pad)
    x2p = min(w, int(x2) + pad)
    y2p = min(h, int(y2) + pad)

    crop = img[y1p:y2p, x1p:x2p]
    if crop.size == 0:
        result_label.config(text="❌ Crop failed!", fg="red")
        return

    # OCR using util
    text_out, conf = read_license_plate(crop)
    if not text_out:
        text_out = "Not Found"
        conf = 0.0

    # Save unique
    saved_img_path, saved_text_flag = save_plate_and_text(crop, text_out, conf, selected_image_path, f"{x1p},{y1p},{x2p},{y2p}")

    # Update GUI
    result_label.config(text=f"Detected: {text_out}  (conf: {conf:.2f})", fg="lime")
    # show cropped plate
    crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).resize((300, 150))
    crop_tk = ImageTk.PhotoImage(crop_pil)
    crop_panel.configure(image=crop_tk)
    crop_panel.image = crop_tk

    # show saved info
    info = []
    if saved_img_path:
        info.append(f"Saved image: {os.path.basename(saved_img_path)}")
    if saved_text_flag:
        info.append("Text saved")
    if info:
        saved_label.config(text=" | ".join(info), fg="white")
    else:
        saved_label.config(text="Already saved (duplicate)", fg="yellow")


# GUI Layout
tk.Label(root, text="Smart parking system", font=("Arial", 20, "bold"), bg="#1e1e1e", fg="white").pack(pady=8)
tk.Button(root, text="Select Image", font=("Arial", 13), command=select_image, bg="#0078ff", fg="white", width=20).pack(pady=8)
panel = Label(root, bg="#1e1e1e")
panel.pack(pady=4)
tk.Button(root, text="Detect Plate", font=("Arial", 13), command=detect_plate, bg="green", fg="white", width=20).pack(pady=12)
result_label = Label(root, text="", font=("Arial", 14), bg="#1e1e1e", fg="white")
result_label.pack(pady=6)
saved_label = Label(root, text="", font=("Arial", 12), bg="#1e1e1e", fg="white")
saved_label.pack(pady=4)
crop_panel = Label(root, bg="#1e1e1e")
crop_panel.pack(pady=6)
root.mainloop()
