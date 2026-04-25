from ultralytics import YOLO
import os, shutil

model = YOLO("license_plate_detector.pt")

SRC = "Cars"

IMG_OUT = "dataset/images/val"
LBL_OUT = "dataset/labels/val"

os.makedirs(IMG_OUT, exist_ok=True)
os.makedirs(LBL_OUT, exist_ok=True)

count = 0

for img in os.listdir(SRC):
    if img.endswith((".jpg",".png",".jpeg")):
        path = os.path.join(SRC, img)
        result = model(path, conf=0.6)[0]

        if len(result.boxes) > 0:
            shutil.copy(path, IMG_OUT)
            h, w = result.orig_shape

            with open(os.path.join(LBL_OUT, img.replace(".jpg",".txt")), "w") as f:
                for box in result.boxes.xywhn:
                    x,y,bw,bh = box
                    f.write(f"0 {x} {y} {bw} {bh}\n")

            count += 1
        if count == 30:
            break

print("Auto validation dataset ready:", count, "images")
