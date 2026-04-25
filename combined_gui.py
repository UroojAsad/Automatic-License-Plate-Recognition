import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
from util import read_license_plate

# ---- MODEL ----
model = YOLO("license_plate_detector.pt")

# ---- WINDOW ----
root = tk.Tk()
root.title("Automatic License Plate Recognition")
root.geometry("1000x650")
root.configure(bg="#F8FAFC")
root.resizable(False, False)

# ---- COLORS ----
BG = "#F8FAFC"
CARD = "#FFFFFF"
PRIMARY = "#2563EB"
SECONDARY = "#0EA5E9"
TEXT = "#0F172A"
MUTED = "#64748B"
ERROR = "#DC2626"

# ---- FONTS ----
FONT_TITLE = ("Segoe UI", 24, "bold")
FONT_SUB = ("Segoe UI", 11)
FONT_BTN = ("Segoe UI", 12, "bold")
FONT_RES = ("Consolas", 18, "bold")

# ---- STATE ----
selected_image_path = ""

# ---- BUTTON STYLE ----
def styled_button(parent, text, cmd):
    return tk.Button(
        parent, text=text, command=cmd,
        bg=PRIMARY, fg="white",
        font=FONT_BTN,
        relief="flat",
        width=22,
        activebackground=SECONDARY
    )

# ---- IMAGE SELECT ----
def select_image():
    global selected_image_path
    path = filedialog.askopenfilename(
        filetypes=[("Images", "*.jpg *.png *.jpeg")]
    )
    if not path:
        return

    selected_image_path = path
    img = Image.open(path).resize((420, 300))
    img_tk = ImageTk.PhotoImage(img)
    image_panel.config(image=img_tk)
    image_panel.image = img_tk
    status_label.config(text="Image Loaded Successfully", fg=PRIMARY)

# ---- DETECT ----
def detect_plate():
    if not selected_image_path:
        status_label.config(text="Please select an image first", fg=ERROR)
        return

    img = cv2.imread(selected_image_path)
    results = model(img)[0]
    boxes = results.boxes.data.tolist()

    if not boxes:
        status_label.config(text="No License Plate Detected", fg=ERROR)
        return

    x1, y1, x2, y2, _, _ = boxes[0]
    crop = img[int(y1):int(y2), int(x1):int(x2)]

    text, conf = read_license_plate(crop)

    result_label.config(text=text)
    conf_label.config(text=f"Confidence: {conf:.2f}")

    crop_img = Image.fromarray(
        cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    ).resize((300, 140))
    crop_tk = ImageTk.PhotoImage(crop_img)
    crop_panel.config(image=crop_tk)
    crop_panel.image = crop_tk

    status_label.config(text="License Plate Detected Successfully", fg=PRIMARY)

# ---- UI ----
tk.Label(root, text="AUTOMATIC LICENSE PLATE RECOGNITION",
         font=FONT_TITLE, bg=BG, fg=PRIMARY).pack(pady=15)

tk.Label(root, text="Vehicle Number Detection using Deep Learning",
         font=FONT_SUB, bg=BG, fg=MUTED).pack()

main = tk.Frame(root, bg=BG)
main.pack(pady=25)

# ---- LEFT ----
left = tk.Frame(main, bg=CARD, width=450, height=420)
left.grid(row=0, column=0, padx=20)
left.pack_propagate(False)

image_panel = tk.Label(left, bg=CARD)
image_panel.pack(pady=20)

styled_button(left, "Select Image", select_image).pack(pady=10)

# ---- RIGHT ----
right = tk.Frame(main, bg=CARD, width=450, height=420)
right.grid(row=0, column=1, padx=20)
right.pack_propagate(False)

tk.Label(right, text="DETECTED LICENSE PLATE",
         bg=CARD, fg=MUTED).pack(pady=10)

result_label = tk.Label(right, text="---",
                        font=FONT_RES,
                        bg=CARD, fg=TEXT)
result_label.pack()

conf_label = tk.Label(right, text="Confidence: 0.00",
                      bg=CARD, fg=MUTED)
conf_label.pack()

crop_panel = tk.Label(right, bg=CARD)
crop_panel.pack(pady=10)

styled_button(right, "Detect Plate", detect_plate).pack(pady=10)

status_label = tk.Label(root, text="Waiting for Image...",
                        bg=BG, fg=MUTED)
status_label.pack(pady=12)

root.mainloop()
