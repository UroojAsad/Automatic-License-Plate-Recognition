from ultralytics import YOLO
import cv2
import argparse
from util import read_license_plate, write_csv

parser = argparse.ArgumentParser()
parser.add_argument("--image", type=str)
parser.add_argument("--video", type=str)
args = parser.parse_args()

model = YOLO("license_plate_detector.pt")

# =========================
def process_image(path):
    img = cv2.imread(path)
    if img is None:
        print("Image not found")
        return

    results = model(img)[0]
    output = {0: {}}

    for i, box in enumerate(results.boxes.data.tolist()):
        x1,y1,x2,y2,score,cls = box
        crop = img[int(y1):int(y2), int(x1):int(x2)]

        text, conf = read_license_plate(crop)

        if text:
            output[0][i] = {
                "car": {"bbox":[int(x1),int(y1),int(x2),int(y2)]},
                "license_plate":{
                    "bbox":[int(x1),int(y1),int(x2),int(y2)],
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
            crop = frame[int(y1):int(y2), int(x1):int(x2)]

            text, conf = read_license_plate(crop)

            if text:
                results_all.setdefault(frame_id,{})
                results_all[frame_id][i] = {
                    "car":{"bbox":[int(x1),int(y1),int(x2),int(y2)]},
                    "license_plate":{
                        "bbox":[int(x1),int(y1),int(x2),int(y2)],
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
