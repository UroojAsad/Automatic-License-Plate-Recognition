from ultralytics import YOLO
import cv2
import argparse
from util import read_license_plate, write_csv

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
parser.add_argument("--video", type=str)
args = parser.parse_args()

model = YOLO("license_plate_detector.pt")

CONF_THRES = 0.5    # detection filter
OCR_THRES  = 0.4    # OCR confidence filter

# =========================
def preprocess_plate(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    return gray

# =========================
def process_image(path):
    img = cv2.imread(path)
    if img is None:
        print("❌ Image not found")
        return

    results = model(img)[0]
    output = {0: {}}

    for i, box in enumerate(results.boxes.data.tolist()):
        x1,y1,x2,y2,score,cls = box

        if score < CONF_THRES:
            continue

        h, w, _ = img.shape
        pad = 5
        x1 = max(0, int(x1)-pad)
        y1 = max(0, int(y1)-pad)
        x2 = min(w, int(x2)+pad)
        y2 = min(h, int(y2)+pad)

        crop = img[y1:y2, x1:x2]
        crop = preprocess_plate(crop)

        text, conf = read_license_plate(crop)

        if text and conf > OCR_THRES:
            output[0][i] = {
                "car": {"bbox":[x1,y1,x2,y2]},
                "license_plate":{
                    "bbox":[x1,y1,x2,y2],
                    "bbox_score":float(score),
                    "text":text,
                    "text_score":float(conf)
                }
            }

    if output[0]:
        write_csv(output, "image_output.csv")
        print("✅ IMAGE DONE")
    else:
        print("❌ NO PLATE READ")

# =========================
def process_video(path):
    cap = cv2.VideoCapture(path)
    results_all = {}
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = model(frame)[0].boxes.data.tolist()

        for i, box in enumerate(detections):
            x1,y1,x2,y2,score,cls = box
            if score < CONF_THRES:
                continue

            h, w, _ = frame.shape
            pad = 5
            x1 = max(0, int(x1)-pad)
            y1 = max(0, int(y1)-pad)
            x2 = min(w, int(x2)+pad)
            y2 = min(h, int(y2)+pad)

            crop = frame[y1:y2, x1:x2]
            crop = preprocess_plate(crop)

            text, conf = read_license_plate(crop)

            if text and conf > OCR_THRES:
                results_all.setdefault(frame_id,{})
                results_all[frame_id][i] = {
                    "car":{"bbox":[x1,y1,x2,y2]},
                    "license_plate":{
                        "bbox":[x1,y1,x2,y2],
                        "bbox_score":float(score),
                        "text":text,
                        "text_score":float(conf)
                    }
                }

        frame_id += 1

    cap.release()

    if results_all:
        write_csv(results_all, "video_output.csv")
        print("✅ VIDEO DONE")
    else:
        print("❌ NO PLATES FOUND")

# =========================
if __name__ == "__main__":
    if args.image:
        process_image(args.image)
    elif args.video:
        process_video(args.video)
    else:
        print("Use --image or --video")
