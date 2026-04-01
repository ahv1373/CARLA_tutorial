import os
import shutil
from pathlib import Path
import yaml
import torch
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import numpy as np
import time

# Directory setup
BASE_DIR = Path(__file__).parent / "data"  # Set base directory to the local "data" folder
DATA_DIR = BASE_DIR
IMAGES_DIR = DATA_DIR / "rgb"
AUG_IMAGES_DIR = DATA_DIR / "aug_rgb"
LABELS_DIR = DATA_DIR / "yolo_annotation" / "normal"
AUG_LABELS_DIR = DATA_DIR / "yolo_annotation" / "augmented"
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"
WORKING_LABELS_DIR = BASE_DIR / "labels" / "normal"
WORKING_AUG_LABELS_DIR = BASE_DIR / "labels" / "augmented"

# Create directories if they don't exist
for dir_path in [WORKING_LABELS_DIR, WORKING_AUG_LABELS_DIR, OUTPUT_DIR, MODEL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Copy label files to working directories
for file in LABELS_DIR.glob("*.txt"):
    shutil.copy(file, WORKING_LABELS_DIR)
for file in AUG_LABELS_DIR.glob("*.txt"):
    shutil.copy(file, WORKING_AUG_LABELS_DIR)

# Class and weather definitions
CLASSES = {
    114: "Car",           # vehicle.car
    115: "Pedestrian",    # walker.pedestrian
    122: "TrafficLight",  # traffic.traffic_light
    177: "Bus"           # vehicle.bus
}
WEATHER_CONDITIONS = ["Clear_Day", "Clear_Night", "Heavy_Rain", "Foggy"]

def verify_setup():
    """Verify that all required directories exist."""
    required_dirs = [DATA_DIR, IMAGES_DIR, AUG_IMAGES_DIR, LABELS_DIR, AUG_LABELS_DIR]
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")

def fix_labels(labels_dir, valid_classes):
    """Remove invalid class IDs from label files."""
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            lines = f.readlines()

        fixed_lines = []
        for line in lines:
            try:
                class_id = int(line.split()[0])
                if class_id in valid_classes:
                    fixed_lines.append(line)
                else:
                    print(f"Invalid class {class_id} in file {label_file}")
            except (IndexError, ValueError):
                print(f"Invalid line format in {label_file}: {line.strip()}")

        with open(label_file, "w") as f:
            f.writelines(fixed_lines)

def get_weather(img_path):
    """Extract weather condition from image file name."""
    stem = img_path.stem
    parts = stem.split("_")
    if parts[-1].startswith("aug"):
        weather = "_".join(parts[1:-2])  # For augmented data
    else:
        weather = "_".join(parts[1:-1])  # For original data
    return weather

def prepare_dataset():
    """Prepare image-label pairs for training."""
    fix_labels(WORKING_LABELS_DIR, CLASSES.keys())
    fix_labels(WORKING_AUG_LABELS_DIR, CLASSES.keys())

    # Separate original and augmented data
    original_images = list(IMAGES_DIR.glob("*.png"))
    aug_images = list(AUG_IMAGES_DIR.glob("*.png"))
    original_labels = list(WORKING_LABELS_DIR.glob("*.txt"))
    aug_labels = list(WORKING_AUG_LABELS_DIR.glob("*.txt"))

    # Pair original data
    orig_img_keys = {img.stem.replace("rgb_", ""): img for img in original_images}
    orig_lbl_keys = {lbl.stem.replace("seg_", ""): lbl for lbl in original_labels}
    original_pairs = [(orig_img_keys[key], orig_lbl_keys[key])
                      for key in orig_img_keys if key in orig_lbl_keys]

    # Pair augmented data
    aug_img_keys = {img.stem: img for img in aug_images}
    aug_lbl_keys = {lbl.stem: lbl for lbl in aug_labels}
    aug_pairs = [(aug_img_keys[key], aug_lbl_keys[key])
                 for key in aug_img_keys if key in aug_lbl_keys]

    # Combine pairs
    all_pairs = original_pairs + aug_pairs
    if not all_pairs:
        raise ValueError("No matching image-label pairs found!")

    images = [img for img, _ in all_pairs]
    labels = [lbl for _, lbl in all_pairs]
    print(f"Found {len(images)} valid image-label pairs")
    return images, labels

def split_dataset(images, labels):
    """Split dataset into train, validation, and test sets."""
    train_val_imgs, test_imgs, train_val_lbls, test_lbls = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    train_imgs, val_imgs, train_lbls, val_lbls = train_test_split(
        train_val_imgs, train_val_lbls, test_size=0.25, random_state=42
    )

    print(f"Train: {len(train_imgs)}, Validation: {len(val_imgs)}, Test: {len(test_imgs)}")
    return train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, test_lbls

def organize_dataset(train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, test_lbls):
    """Organize dataset into train, val, and test directories."""
    splits = {
        "train": (train_imgs, train_lbls),
        "val": (val_imgs, val_lbls),
        "test": (test_imgs, test_lbls)
    }
    dataset_dir = Path("/dataset")
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
    """Create dataset.yaml file for YOLO training."""
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

def train_model(model_path, data_yaml):
    """Train YOLO model with specified parameters."""
    start_time = time.time()
    model = YOLO(model_path)
    model_name = Path(model_path).stem
    results = model.train(
        data=data_yaml,
        task='segment',  # Explicitly specify segmentation task
        epochs=1,
        imgsz=640,
        batch=8,
        device=0 if torch.cuda.is_available() else "cpu",
        project="/data/models",
        name=f"{model_name}_segment",
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

def evaluate_model(model, test_imgs, weather_conditions, data_yaml):
    """Evaluate model on test set."""
    metrics = {}
    # Perform evaluation on the entire test set
    results = model.val(data=data_yaml, split="test")
    
    # General metrics
    overall_metrics = {
        "mAP50": results.seg.map50 if hasattr(results, 'seg') else 0.0,
        "mAP50-95": results.seg.map if hasattr(results, 'seg') else 0.0,
        "inference_time": results.speed["inference"],
        "per_class_map": results.seg.ap_class_by_class if hasattr(results, 'seg') else []
    }
    
    # Assign metrics to each weather condition (temporary until per-condition evaluation is implemented)
    for condition in weather_conditions:
        condition_imgs = [img for img in test_imgs if get_weather(img) == condition]
        if condition_imgs:
            metrics[condition] = overall_metrics
        else:
            metrics[condition] = {"note": "No images for this condition"}
    
    return metrics

def save_results(model_name, metrics, training_time):
    """Save training and evaluation results to a file."""
    results_file = OUTPUT_DIR / f"{model_name}_detailed_results.txt"
    with open(results_file, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n\n")
        f.write("Performance Across Weather Conditions:\n")
        f.write("-" * 50 + "\n")
        
        for condition, condition_metrics in metrics.items():
            f.write(f"\nWeather Condition: {condition}\n")
            if "note" in condition_metrics:
                f.write(f"{condition_metrics['note']}\n")
            else:
                f.write(f"mAP50: {condition_metrics['mAP50']:.4f}\n")
                f.write(f"mAP50-95: {condition_metrics['mAP50-95']:.4f}\n")
                f.write(f"Inference Time: {condition_metrics['inference_time']:.4f} ms\n")
                f.write("\nPer-Class Performance:\n")
                for class_id, class_map in enumerate(condition_metrics['per_class_map']):
                    f.write(f"{CLASSES[class_id]}: {class_map:.4f}\n")
            f.write("-" * 50 + "\n")

def main():
    """Main function to prepare dataset, train, and evaluate models."""
    try:
        verify_setup()
        
        # Prepare and split dataset
        images, labels = prepare_dataset()
        train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, test_lbls = split_dataset(images, labels)
        
        # Organize dataset and create YAML
        dataset_dir = organize_dataset(train_imgs, val_imgs, test_imgs, train_lbls, val_lbls, test_lbls)
        data_yaml = create_dataset_yaml(dataset_dir)
        
        # Model paths
        model_paths = [
            "yolo11n-seg.pt",
            "yolo11m-seg.pt",
            "yolo11l-seg.pt"
        ]

        for model_path in model_paths:
            try:
                print(f"\nTraining Model: {model_path}")
                model, results, training_time = train_model(model_path, data_yaml)
        
                print(f"Evaluating Model: {model_path}")
                metrics = evaluate_model(model, test_imgs, WEATHER_CONDITIONS, data_yaml)
                save_results(Path(model_path).stem, metrics, training_time)
        
            except Exception as e:
                print(f"Error with {model_path}: {e}")
                if 'metrics' in locals():
                    print(f"\nResults Summary for {model_path}:")
                    print("-" * 50)
                    for condition, condition_metrics in metrics.items():
                        print(f"\nWeather Condition: {condition}")
                        if "note" not in condition_metrics:
                            print(f"mAP50: {condition_metrics['mAP50']:.4f}")
                            print(f"mAP50-95: {condition_metrics['mAP50-95']:.4f}")
                            print(f"Inference Time: {condition_metrics['inference_time']:.2f} ms")
                    print(f"\nTotal Training Time: {training_time:.2f} seconds")
                    print("-" * 50)
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()