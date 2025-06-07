import os
import shutil
from pathlib import Path
import yaml
import torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import numpy as np
import time
from PIL import Image  


ROOT_DIR = Path().absolute()
DATA_DIR = Path(os.path.join(os.path.dirname(__file__), "data"))
IMAGES_DIR = DATA_DIR / "rgb"
AUG_IMAGES_DIR = DATA_DIR / "aug_rgb"
LABELS_DIR = DATA_DIR / "yolo_annotation/normal"
AUG_LABELS_DIR = DATA_DIR / "yolo_annotation/augmented"
OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = ROOT_DIR / "models"


for dir_path in [OUTPUT_DIR, MODEL_DIR]:
    dir_path.mkdir(exist_ok=True)


CLASSES = {0: "Pedestrian", 1: "Traffic Light", 2: "Car", 3: "Bus"}
WEATHER_CONDITIONS = ["clear", "foggy", "rainy", "snowy"]

def verify_setup():
    
    required_dirs = [DATA_DIR, IMAGES_DIR, AUG_IMAGES_DIR, LABELS_DIR, AUG_LABELS_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

def fix_labels(labels_dir, valid_classes):
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            class_id = int(line.split()[0])
            if class_id in valid_classes:
                fixed_lines.append(line)
            else:
                print(f"Invalid class {class_id} in file {label_file}")

        
        with open(label_file, "w") as f:
            f.writelines(fixed_lines)


fix_labels(LABELS_DIR, range(len(CLASSES)))
fix_labels(AUG_LABELS_DIR, range(len(CLASSES)))

def prepare_dataset():
    fix_labels(LABELS_DIR, CLASSES.keys())
    fix_labels(AUG_LABELS_DIR, CLASSES.keys())

    images = list(IMAGES_DIR.glob("*.png")) + list(AUG_IMAGES_DIR.glob("*.png"))
    labels = list(LABELS_DIR.glob("*.txt")) + list(AUG_LABELS_DIR.glob("*.txt"))

    image_stems = {img.stem.replace("rgb_", ""): img for img in images}
    label_stems = {lbl.stem.replace("rgb_", ""): lbl for lbl in labels}

    paired_images = []
    paired_labels = []
    for stem in image_stems:
        if stem in label_stems:
            paired_images.append(image_stems[stem])
            paired_labels.append(label_stems[stem])
        else:
            print(f"Warning: No matching label for image {image_stems[stem]}")

    if not paired_images:
        raise ValueError("No matching image-label pairs found!")

    print(f"Found {len(paired_images)} valid image-label pairs")
    return paired_images, paired_labels

def split_dataset(images, labels):
    
    train_val_imgs, test_imgs, train_val_lbls, test_lbls = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        train_val_imgs, train_val_lbls, test_size=0.25, random_state=42  
    )

    print(f"Train: {len(train_imgs)}, Validation: {len(val_imgs)}, Test: {len(test_imgs)}")
    return train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, test_lbls

def organize_dataset(train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, test_lbls):
    
    splits = {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls)
    }
    dataset_dir = DATA_DIR / "dataset"
    for split, (imgs, lbls) in splits.items():
        img_dir = dataset_dir / "images" / split
        lbl_dir = dataset_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img, lbl in zip(imgs, lbls):
            shutil.copy(img, img_dir / img.name)
            shutil.copy(lbl, lbl_dir / lbl.name)

    return dataset_dir

def create_dataset_yaml(dataset_dir):
    yaml_data = {
        "path": str(dataset_dir.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(CLASSES),
        "names": list(CLASSES.values())
    }
    yaml_path = dataset_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(yaml_data, f, sort_keys=False)
    return yaml_path

def train_model(model_type, data_yaml):
    start_time = time.time()
    model = YOLO(model_type)
    results = model.train(
        data=data_yaml,
        epochs=5,
        imgsz=640,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        project=str(OUTPUT_DIR),
        name=f"{model_type.split('.')[0]}_detect",  
        exist_ok=True,
        pretrained=True,
        verbose=True,
        patience=10,
        lr0=0.001,
        optimizer="AdamW",
        plots=True
    )
    training_time = time.time() - start_time
    return model, results, training_time

def evaluate_model(model, test_imgs, weather_conditions):
    metrics = {}
    for condition in weather_conditions:
        
        temp_dir = DATA_DIR / "temp" / condition
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        
        condition_imgs = [img for img in test_imgs if condition in str(img)]
        for img in condition_imgs:
            shutil.copy(img, temp_dir / img.name)
        
        
        results = model.val(
            data=str(DATA_DIR / "dataset" / "dataset.yaml"),
            split="test"
        )
        
        
        shutil.rmtree(temp_dir)
        
        metrics[condition] = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "inference_time": results.speed["inference"],
            "per_class_map": results.box.ap_class_by_class
        }
    return metrics

def save_results(model_name, metrics, training_time):
    results_file = OUTPUT_DIR / f"{model_name}_detailed_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n\n")
        f.write("Performance Across Weather Conditions:\n")
        f.write("-" * 50 + "\n")
        
        for condition, condition_metrics in metrics.items():
            f.write(f"\nWeather Condition: {condition}\n")
            f.write(f"mAP50: {condition_metrics['mAP50']:.4f}\n")
            f.write(f"mAP50-95: {condition_metrics['mAP50-95']:.4f}\n")
            f.write(f"Inference Time: {condition_metrics['inference_time']:.4f} ms\n")
            
            f.write("\nPer-Class Performance:\n")
            for class_id, class_map in enumerate(condition_metrics['per_class_map']):
                f.write(f"{CLASSES[class_id]}: {class_map:.4f}\n")
            f.write("-" * 50 + "\n")

def fix_coordinates(coords, img_width, img_height):
    """Helper function to fix and validate coordinates"""
    x_center, y_center = coords[0], coords[1]
    width, height = coords[2], coords[3]
    
    # Ensure coordinates are within image bounds
    x_center = max(0, min(x_center, img_width))
    y_center = max(0, min(y_center, img_height))
    width = max(0, min(width, img_width - x_center))
    height = max(0, min(height, img_height - y_center))
    
    # Normalize coordinates
    x_norm = x_center / img_width
    y_norm = y_center / img_height
    w_norm = width / img_width
    h_norm = height / img_height
    
    return x_norm, y_norm, w_norm, h_norm

def convert_npy_to_txt(npy_dir, output_dir, class_id=0):
    output_dir.mkdir(parents=True, exist_ok=True)
    for npy_file in npy_dir.glob("*.npy"):
        data = np.load(npy_file)
        txt_file = output_dir / f"{npy_file.stem.replace('seg_', 'rgb_')}.txt"
        
        # Read corresponding image to get dimensions
        img_file = IMAGES_DIR / f"{npy_file.stem.replace('seg_', 'rgb_')}.png"
        if not img_file.exists():
            print(f"Warning: No matching image found for {npy_file}")
            continue
            
        from PIL import Image
        with Image.open(img_file) as img:
            img_width, img_height = img.size
        
        with open(txt_file, "w") as f:
            for obj in data:
                if len(obj) >= 4:
                    # Use the helper function to fix coordinates
                    x_norm, y_norm, w_norm, h_norm = fix_coordinates(
                        obj[:4], img_width, img_height
                    )
                    if all(0 <= coord <= 1 for coord in [x_norm, y_norm, w_norm, h_norm]):
                        f.write(f"{class_id} {x_norm:.6f} {y_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n")
                    else:
                        print(f"Warning: Invalid normalized coordinates in {npy_file}")
                else:
                    print(f"Warning: Invalid data format in {npy_file}")


def main():
    try:
        verify_setup()
        convert_npy_to_txt(Path(DATA_DIR / "semantics"), LABELS_DIR)
     
        images, labels = prepare_dataset()
        train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, test_lbls = split_dataset(images, labels)
        
        dataset_dir = organize_dataset(train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, test_lbls)
        data_yaml = create_dataset_yaml(dataset_dir)
       
        model_types = ["yolov8n.pt", "yolov8m.pt", "yolov8l.pt"]
        for model_type in model_types:
            try:
                print(f"\nModel Training: {model_type}")
                model, results, training_time = train_model(model_type, data_yaml)
                
                print(f"Model Evaluation: {model_type}")
                metrics = evaluate_model(model, test_imgs, WEATHER_CONDITIONS)
                save_results(model_type.split(".")[0], metrics, training_time)
                
                print(f"\nResults Summary for {model_type}:")
                print("-" * 50)
                for condition, condition_metrics in metrics.items():
                    print(f"\nWeather Condition: {condition}")
                    print(f"mAP50: {condition_metrics['mAP50']:.4f}")
                    print(f"mAP50-95: {condition_metrics['mAP50-95']:.4f}")
                    print(f"Inference Time: {condition_metrics['inference_time']:.2f} ms")
                print(f"\nTotal Training Time: {training_time:.2f} seconds")
                print("-" * 50)
            
            except Exception as e:
                print(f"Error with {model_type}: {e}")
                continue
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()