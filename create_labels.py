import os

# Paths
image_dirs = {
    "train": "dataset/images/train",
    "val": "dataset/images/val"
}

label_dirs = {
    "train": "dataset/labels/train",
    "val": "dataset/labels/val"
}

# Ensure label directories exist
for key in label_dirs:
    os.makedirs(label_dirs[key], exist_ok=True)

# Function to create dummy YOLO labels if missing
def create_yolo_labels(image_path, label_path):
    image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    for img_file in image_files:
        label_file = os.path.splitext(img_file)[0] + ".txt"
        label_full_path = os.path.join(label_path, label_file)
        if not os.path.exists(label_full_path) or os.path.getsize(label_full_path) == 0:
            # Creating dummy label with one box covering whole image (x_center=0.5, y_center=0.5, w=1, h=1)
            with open(label_full_path, "w") as f:
                f.write("0 0.5 0.5 1 1\n")  # class 0, normalized coordinates

# Create labels for train and val
for key in image_dirs:
    create_yolo_labels(image_dirs[key], label_dirs[key])

print("✅ YOLO labels ready for train and val folders!")
