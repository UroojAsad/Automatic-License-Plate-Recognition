import os
import re
import cv2
import easyocr
import numpy as np
import pytesseract
from typing import Tuple

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

easy_reader = easyocr.Reader(['en'], gpu=False)

CHAR_FIX = {
    "O": "0", "I": "1", "Z": "2",
    "S": "5", "B": "8"
}

def fix_common_mistakes(text):
    return "".join(CHAR_FIX.get(c, c) for c in text)

# -------------------------
# EASYOCR PREPROCESS
# -------------------------
def preprocess_easy(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    return gray

# -------------------------
# TESSERACT PREPROCESS
# -------------------------
def preprocess_tess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

# -------------------------
# CLEAN TEXT (FLEXIBLE)
# -------------------------
def clean_plate_text(text):
    if not text:
        return None

    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    # ---------- DIGIT GROUP (BOTTOM LINE) ----------
    digit_groups = re.findall(r"\d{3,}", text)
    if not digit_groups:
        return None

    number_part = max(digit_groups, key=len)  # longest digits

    # ---------- LETTER GROUP (TOP LINE) ----------
    text_wo_digits = text.replace(number_part, "")
    letter_groups = re.findall(r"[A-Z]{2,}", text_wo_digits)
    if not letter_groups:
        return None

    letter_part = letter_groups[0][:3]

    return f"{letter_part}-{number_part[:4]}"



# -------------------------
# OCR READ
# -------------------------
def read_license_plate(plate_img: np.ndarray) -> Tuple[str, float]:

    if plate_img is None or plate_img.size == 0:
        return None, 0.0

    # ---- EasyOCR FIRST ----
    try:
        img1 = preprocess_easy(plate_img)
        res1 = easy_reader.readtext(img1, detail=1)
        if res1:
            txt = " ".join([r[1] for r in res1])
            conf = max([r[2] for r in res1])
            clean = clean_plate_text(txt)
            if clean:
                return clean, float(conf)
    except:
        pass

    # ---- TESSERACT BACKUP ----
    try:
        img2 = preprocess_tess(plate_img)
        config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        txt = pytesseract.image_to_string(img2, config=config)
        clean = clean_plate_text(txt)
        if clean:
            return clean, 0.60
    except:
        pass

    return None, 0.0
