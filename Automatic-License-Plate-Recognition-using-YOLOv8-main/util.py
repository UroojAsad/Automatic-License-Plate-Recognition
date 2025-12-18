# util.py
import os
import cv2
import easyocr
import numpy as np
from typing import Tuple

reader = easyocr.Reader(['en'], gpu=False)

PROVINCES = [
    "PUNJAB","SINDH","BALOCHISTAN","BALOCHISTÁN",
    "KPK","KP","ICT","ISLAMABAD",
    "GILGIT","BALTISTAN","QUETTA","LAHORE"
]

# -------- SMART FIX (ONLY FOR NUMBERS PART) --------
LETTER_TO_DIGIT = {
    "O": "0",
    "I": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6"
}

def fix_plate_format(text: str) -> str:
    """
    Format: LETTERS + DIGITS
    Example: LEA1856, BT43288
    """
    if len(text) < 4:
        return text

    text = text.upper()

    # split letters and numbers
    letters = ""
    numbers = ""

    for c in text:
        if c.isalpha() and len(numbers) == 0:
            letters += c
        elif c.isdigit() or c.isalpha():
            numbers += c

    # fix only number part
    fixed_numbers = ""
    for c in numbers:
        if c.isalpha():
            fixed_numbers += LETTER_TO_DIGIT.get(c, c)
        else:
            fixed_numbers += c

    return letters + fixed_numbers


def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # resize (helps blur)
    h, w = gray.shape
    gray = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # sharpen
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)

    return sharp


def read_license_plate(plate_img: np.ndarray) -> Tuple[str, float]:

    if plate_img is None or plate_img.size == 0:
        return None, 0.0

    proc = preprocess_plate(plate_img)

    results = reader.readtext(proc, detail=1)

    if not results:
        return None, 0.0

    texts = [r[1].upper().strip() for r in results]
    conf = max([r[2] for r in results])

    full = " ".join(texts)

    # remove province/city
    for p in PROVINCES:
        full = full.replace(p, " ")

    full = " ".join(full.split())

    # keep only A-Z 0-9
    clean = "".join([c for c in full if c.isalnum()])

    clean = fix_plate_format(clean)

    if len(clean) < 4:
        return None, 0.0

    return clean, float(conf)


def write_csv(results, output_path):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("frame,car_id,plate_text,confidence\n")

        for frame in results:
            for cid in results[frame]:
                lp = results[frame][cid]["license_plate"]
                f.write(f"{frame},{cid},{lp['text']},{lp['text_score']}\n")
