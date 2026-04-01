import os
import random
import shutil
import xml.etree.ElementTree as ET

# --- Configuration ---
# Make sure these paths match where you extracted your zip file
RAW_IMAGES_DIR = 'raw_data/images'
RAW_ANNOTATIONS_DIR = 'raw_data/annotations'
OUTPUT_DIR = 'datasets'
SPLIT_RATIO = 0.8 # 80% for training, 20% for validation
# ---------------------

# 1. Create target directories if they don't exist
for split in ['train', 'valid']:
    for category in ['images', 'labels']:
        os.makedirs(os.path.join(OUTPUT_DIR, split, category), exist_ok=True)

# 2. Get list of all XML files and shuffle them
xml_files = [f for f in os.listdir(RAW_ANNOTATIONS_DIR) if f.endswith('.xml')]
random.shuffle(xml_files)

# 3. Split the data
split_index = int(len(xml_files) * SPLIT_RATIO)
train_files = xml_files[:split_index]
valid_files = xml_files[split_index:]

# This list will automatically discover your object classes
classes = []

def convert_bbox(size, box):
    """Converts Pascal VOC (xmin, xmax, ymin, ymax) to YOLO (x_center, y_center, width, height)"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    return (x * dw, y * dh, w * dw, h * dh)

def process_files(files, split_name):
    for xml_file in files:
        tree = ET.parse(os.path.join(RAW_ANNOTATIONS_DIR, xml_file))
        root = tree.getroot()

        # Get image size
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        # Find matching image file (checks common extensions)
        base_name = os.path.splitext(xml_file)[0]
        image_file = None
        for ext in ['.jpg', '.png', '.jpeg', '.JPG']:
            if os.path.exists(os.path.join(RAW_IMAGES_DIR, base_name + ext)):
                image_file = base_name + ext
                break
        
        if not image_file:
            print(f"Warning: Image for {xml_file} not found. Skipping.")
            continue

        image_path = os.path.join(RAW_IMAGES_DIR, image_file)
        txt_path = os.path.join(OUTPUT_DIR, split_name, 'labels', base_name + '.txt')

        # Convert XML to TXT
        with open(txt_path, 'w') as out_file:
            for obj in root.iter('object'):
                difficult = obj.find('difficult')
                if difficult is not None and int(difficult.text) == 1:
                    continue
                
                cls_name = obj.find('name').text
                if cls_name not in classes:
                    classes.append(cls_name)
                cls_id = classes.index(cls_name)

                xmlbox = obj.find('bndbox')
                b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
                bb = convert_bbox((w, h), b)

                out_file.write(f"{cls_id} {' '.join([str(a) for a in bb])}\n")

        # Copy the image to the new folder
        shutil.copy(image_path, os.path.join(OUTPUT_DIR, split_name, 'images', image_file))

print("Processing training data...")
process_files(train_files, 'train')

print("Processing validation data...")
process_files(valid_files, 'valid')

# Save a classes.txt file so you know what ID corresponds to what object
with open(os.path.join(OUTPUT_DIR, 'classes.txt'), 'w') as f:
    for c in classes:
        f.write(c + '\n')

print(f"\nSuccess! Data split and converted. Found {len(classes)} classes: {classes}")