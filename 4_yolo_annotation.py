import os
import numpy as np
import cv2

# Directory setup
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
SEMANTICS_DIRS = [os.path.join(DATA_DIR, "semantics"), os.path.join(DATA_DIR, "aug_semantics")]
RGB_DIRS = {
    "semantics": os.path.join(DATA_DIR, "rgb"),
    "aug_semantics": os.path.join(DATA_DIR, "aug_rgb")
}

ANNOT_DIR_NORMAL = os.path.join(DATA_DIR, "yolo_annotation/normal")
ANNOT_DIR_AUG = os.path.join(DATA_DIR, "yolo_annotation/augmented")
os.makedirs(ANNOT_DIR_NORMAL, exist_ok=True)
os.makedirs(ANNOT_DIR_AUG, exist_ok=True)

# Valid class mappings
valid_classes = {
    114: "Car",           # vehicle.car
    115: "Pedestrian",    # walker.pedestrian
    122: "TrafficLight",  # traffic.traffic_light
    177: "Bus"           # vehicle.bus
}

def generate_yolo_v11_annotations(instance_array, image_shape):
    """
    Generate YOLOv11 annotations from instance segmentation mask.
    
    Args:
        instance_array: 3-channel array (R: class ID, G and B: instance ID)
        image_shape: Shape of the corresponding RGB image (height, width, channels)
    
    Returns:
        List of annotation lines in YOLO format
    """
    height, width = image_shape[:2]
    annotations = []

    # Extract class and instance IDs from the three-channel mask
    class_ids = instance_array[:, :, 0]
    instance_ids = (instance_array[:, :, 1].astype(np.uint16) << 8) + instance_array[:, :, 2]

    # Find unique instance IDs (excluding 0, which is background)
    unique_instance_ids = np.unique(instance_ids)
    unique_instance_ids = unique_instance_ids[unique_instance_ids != 0]

    for inst_id in unique_instance_ids:
        # Create binary mask for this instance
        binary_mask = (instance_ids == inst_id).astype(np.uint8)
        
        # Find external contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Select the largest contour by area
            largest_contour = max(contours, key=cv2.contourArea)
            
            if len(largest_contour) >= 3 and cv2.contourArea(largest_contour) > 50:
                # Get class ID (all pixels in an instance should share the same class)
                cls_id = class_ids[binary_mask > 0][0]
                
                if cls_id in valid_classes:
                    # Normalize contour points
                    normalized_pts = [(float(x) / width, float(y) / height) for [[x, y]] in largest_contour]
                    # Create annotation line
                    line = f"{cls_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized_pts)
                    annotations.append(line)

    return annotations

def process_directory(sem_dir, rgb_dir, is_aug=False):
    """
    Process a directory of semantic masks to generate YOLO annotations.
    
    Args:
        sem_dir: Path to semantics directory
        rgb_dir: Path to corresponding RGB directory
        is_aug: Boolean indicating if processing augmented data
    """
    for fname in os.listdir(sem_dir):
        if not fname.endswith(".npy"):
            continue

        sem_path = os.path.join(sem_dir, fname)
        stem = os.path.splitext(fname)[0]
        
        if is_aug:
            # For augmented data, mask file name like "rgb_something_aug0.npy"
            img_name = stem + ".png"  # Corresponding image is "rgb_something_aug0.png"
            label_path = os.path.join(ANNOT_DIR_AUG, stem + ".txt")
        else:
            # For normal data, mask file name like "seg_something.npy"
            img_name = "rgb_" + stem.replace("seg_", "") + ".png"
            label_path = os.path.join(ANNOT_DIR_NORMAL, stem + ".txt")
        
        img_path = os.path.join(rgb_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        # Load three-channel instance segmentation mask
        sem_mask = np.load(sem_path)
        image = cv2.imread(img_path)
        annotations = generate_yolo_v11_annotations(sem_mask, image.shape)

        # Always write annotation file, even if empty
        with open(label_path, "w") as f:
            for line in annotations:
                f.write(line + "\n")
        
        if annotations:
            print(f"[✔] Wrote: {label_path} with {len(annotations)} annotations")
        else:
            print(f"[✔] Empty annotation file written for: {fname}")

def main():
    """Process all semantics directories to generate YOLOv11 annotations."""
    for sem_path in SEMANTICS_DIRS:
        base = os.path.basename(sem_path)
        is_aug = "aug" in base
        rgb_path = RGB_DIRS[base]
        process_directory(sem_path, rgb_path, is_aug=is_aug)
    print("✅ YOLOv11-compatible annotations generated.")

if __name__ == "__main__":
    main()