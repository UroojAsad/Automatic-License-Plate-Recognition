import os

image_folder = "dataset/images/val"
label_folder = "dataset/labels/val"

# Make sure label folder exists
os.makedirs(label_folder, exist_ok=True)

# Loop through all images
for img_file in os.listdir(image_folder):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        txt_file = os.path.splitext(img_file)[0] + ".txt"
        txt_path = os.path.join(label_folder, txt_file)
        # Create empty label file if it doesn't exist
        if not os.path.exists(txt_path):
            open(txt_path, 'w').close()

print("Temporary empty labels created for validation!")
