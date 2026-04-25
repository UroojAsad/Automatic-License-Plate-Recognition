import os
import shutil
import random

# Correct path to your original images folder (make sure folder exists)
source_folder = r"D:\5th sem project\dip\Automatic-License-Plate-Recognition-using-YOLOv8-main\Cars"

# Paths for train and val folders
train_folder = r"D:\5th sem project\dip\Automatic-License-Plate-Recognition-using-YOLOv8-main\images\train"
val_folder = r"D:\5th sem project\dip\Automatic-License-Plate-Recognition-using-YOLOv8-main\images\val"

# Create train/val folders if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# Get all image files
images = [f for f in os.listdir(source_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle the list randomly
random.shuffle(images)

# Calculate split index (80% train, 20% val)
split_index = int(len(images) * 0.8)

train_images = images[:split_index]
val_images = images[split_index:]

# Move images to respective folders
for img in train_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(train_folder, img))

for img in val_images:
    shutil.copy(os.path.join(source_folder, img), os.path.join(val_folder, img))

print(f"Total images: {len(images)}")
print(f"Training images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")
print("Split done successfully!")
