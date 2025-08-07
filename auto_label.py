import os
import shutil
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Define paths
dataset_path = 'gun_images'
output_path = 'auto_label'
yolo_model_name = 'runs/detect/train1/weights/best.pt' # Using a powerful pre-trained model for best results

# Create output directories
os.makedirs(os.path.join(output_path, 'images/train'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'images/val'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'labels/train'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'labels/val'), exist_ok=True)

# Load a pre-trained YOLO model for auto-labeling
print(f"Loading pre-trained YOLO model: {yolo_model_name}...")
model = YOLO(yolo_model_name)

# --- Configuration ---
# You can adjust these settings to control the auto-labeling process.
# A higher confidence threshold will result in fewer, but more accurate labels.
CONF_THRESHOLD = 0.5
# The classes you want to detect. You should confirm these with the YOLOv8 model's class list.
# For example, 'cell phone', 'bottle', 'remote', etc.
target_classes = [
    'gun', 'knife', 'wrench'
]
# We map the target class names to a simple 0-indexed ID for the YOLO format.
class_map = {class_name: i for i, class_name in enumerate(target_classes)}

# Let's get the YOLO model's class names to find the IDs
yolo_model_class_names = model.names
print("Pre-trained model classes:", yolo_model_class_names)
print("Target classes:", target_classes)

# Map our target classes to the pre-trained model's class IDs
yolo_class_ids_to_target_ids = {}
for i, name in yolo_model_class_names.items():
    if name in target_classes:
        yolo_class_ids_to_target_ids[i] = class_map[name]
        
if not yolo_class_ids_to_target_ids:
    print("Warning: None of the target classes were found in the pre-trained model.")
    print("Please check the 'target_classes' list and the model's class names.")
    exit()

print("Mapping pre-trained model IDs to target IDs:", yolo_class_ids_to_target_ids)

# List of all images to process
all_image_paths = []
for root, dirs, files in os.walk(dataset_path):
    for filename in files:
        if filename.endswith('.png'):
            all_image_paths.append(os.path.join(root, filename))

if not all_image_paths:
    print("Error: No images found in the dataset path. Please check the path.")
    exit()

# --- Auto-labeling process ---
print(f"Found {len(all_image_paths)} images. Starting auto-labeling...")
# This list will store dictionaries of image paths and their generated labels.
image_label_pairs = [] 

for image_path in tqdm(all_image_paths, desc="Processing images"):
    results = model.predict(image_path, conf=CONF_THRESHOLD, verbose=False)
    
    # Process results and write to a YOLO-format label file
    yolo_labels = []
    
    for result in results:
        # Check if any detections were made
        if result.boxes:
            boxes = result.boxes.xyxyn.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i in range(len(boxes)):
                box = boxes[i]
                class_id_yolo = int(classes[i])
                
                # Check if this class is one of our target classes
                if class_id_yolo in yolo_class_ids_to_target_ids:
                    target_class_id = yolo_class_ids_to_target_ids[class_id_yolo]
                    x_min, y_min, x_max, y_max = box
                    
                    # YOLO format: class_id center_x center_y width height
                    center_x = (x_min + x_max) / 2
                    center_y = (y_min + y_max) / 2
                    width = x_max - x_min
                    height = y_max - y_min
                    
                    yolo_labels.append(f'{target_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}')
    
    # Only save the image and label if at least one object was detected
    if yolo_labels:
        image_label_pairs.append({'image_path': image_path, 'labels': yolo_labels})

# Split the data into training and validation sets (80/20 split)
if not image_label_pairs:
    print("No images with detected objects were found. Please check your `target_classes` list or lower the `CONF_THRESHOLD`.")
else:
    print(f"\nFound {len(image_label_pairs)} images with labels. Now splitting into train/val sets.")
    train_pairs, val_pairs = train_test_split(image_label_pairs, test_size=0.2, random_state=42)

    # A final function to copy the files to the new dataset directories
    def copy_files(pairs, image_dir, label_dir):
        for pair in pairs:
            img_path = pair['image_path']
            labels = pair['labels']
            filename = os.path.basename(img_path)
            label_filename = filename.replace('.png', '.txt')
            
            shutil.copy(img_path, os.path.join(image_dir, filename))
            
            with open(os.path.join(label_dir, label_filename), 'w') as f:
                for line in labels:
                    f.write(line + '\n')

    print("\nCopying training files...")
    copy_files(train_pairs, os.path.join(output_path, 'images/train'), os.path.join(output_path, 'labels/train'))
    print("Copying validation files...")
    copy_files(val_pairs, os.path.join(output_path, 'images/val'), os.path.join(output_path, 'labels/val'))
    print("\nAuto-labeling and data preparation complete! Your dataset is ready for training in the 'auto_labeled_dataset' directory.")
