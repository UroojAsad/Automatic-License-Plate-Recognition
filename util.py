import cv2
import re
import numpy as np
import easyocr
import pytesseract
from typing import Tuple

# ---- TESSERACT PATH ----
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

reader = easyocr.Reader(['en'], gpu=False)

# ---- COMMON FIX ----
CHAR_FIX = {
    "O": "0", "I": "1", "Z": "2",
    "S": "5", "B": "8"
}

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    _, th = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

# ---- SMART CLEAN (NO NOT FOUND) ----
def clean_plate_text(text):
    if not text:
        return None

    text = text.upper()
    text = "".join(CHAR_FIX.get(c, c) for c in text)
    text = re.sub(r"[^A-Z0-9 ]", " ", text)

    parts = text.split()
    letters = ""
    digits = ""

    for p in parts:
        if not letters:
            l = re.findall(r"[A-Z]{2,3}", p)
            if l:
                letters = l[0]

        d = re.findall(r"\d{2,5}", p)
        if d:
            digits = d[-1]

    if letters and digits:
        return f"{letters}-{digits}"
    if digits:
        return f"UNK-{digits}"
    if letters:
        return f"{letters}-0000"

    return None

# ---- MAIN OCR ----
def read_license_plate(img: np.ndarray) -> Tuple[str, float]:

    if img is None or img.size == 0:
        return "UNREADABLE", 0.30

    # ---- EASY OCR ----
    try:
        p = preprocess(img)
        res = reader.readtext(p, detail=1)
        if res:
            txt = " ".join([r[1] for r in res])
            conf = max([r[2] for r in res])
            clean = clean_plate_text(txt)
            if clean:
                return clean, float(conf)
    except:
        pass

    # ---- TESSERACT BACKUP ----
    try:
        txt = pytesseract.image_to_string(
            img,
            config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        clean = clean_plate_text(txt)
        if clean:
            return clean, 0.60
    except:
        pass

    # ---- FINAL FALLBACK ----
    return "UNREADABLE", 0.30
