import os
import numpy as np
import cv2

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

valid_classes = {0: "Pedestrian", 1: "Traffic Light", 2: "Car", 3: "Bus"}

def generate_yolo_v11_annotations(instance_array, image_shape):
    height, width = image_shape[:2]
    annotations = []

    for cls_id in valid_classes:
        binary_mask = (instance_array == cls_id).astype(np.uint8)
        if not binary_mask.any():
            continue

        n_labels, labels = cv2.connectedComponents(binary_mask, connectivity=8)

        for inst_id in range(1, n_labels):
            inst_mask = (labels == inst_id).astype(np.uint8)
            contours, _ = cv2.findContours(inst_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if len(cnt) < 3 or cv2.contourArea(cnt) < 50:
                    continue

                normalized_pts = [(x / width, y / height) for [[x, y]] in cnt]
                line = f"{cls_id} " + " ".join(f"{x:.6f} {y:.6f}" for x, y in normalized_pts)
                annotations.append(line)

    return annotations

def process_directory(sem_dir, rgb_dir, is_aug=False):
    for fname in os.listdir(sem_dir):
        if not fname.endswith(".npy"):
            continue

        sem_path = os.path.join(sem_dir, fname)
        stem = os.path.splitext(fname)[0]
        
        
        if is_aug:
            img_name = stem.replace("seg_", "") + ".png"
            label_path = os.path.join(ANNOT_DIR_AUG, stem + ".txt")  
        else:
            img_name = "rgb_" + stem.replace("seg_", "") + ".png"
            label_path = os.path.join(ANNOT_DIR_NORMAL, stem + ".txt")  
        
        img_path = os.path.join(rgb_dir, img_name)

        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue

        sem_mask = np.load(sem_path)
        image = cv2.imread(img_path)
        annotations = generate_yolo_v11_annotations(sem_mask, image.shape)

        if annotations:
            with open(label_path, "w") as f:
                for line in annotations:
                    f.write(line + "\n")
            print(f"[✔] Wrote: {label_path}")
        else:
            if os.path.exists(label_path):
                os.remove(label_path)
            print(f"[✘] No valid annotations for: {fname}")

def main():
    for sem_path in SEMANTICS_DIRS:
        base = os.path.basename(sem_path)
        is_aug = "aug" in base
        rgb_path = RGB_DIRS[base]
        process_directory(sem_path, rgb_path, is_aug=is_aug)
    print("✅ YOLOv11-compatible annotations generated.")

if __name__ == "__main__":
    main()
