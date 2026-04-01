import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

# Paths to your data directories
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RGB_DIR = os.path.join(DATA_DIR, "rgb")
SEM_DIR = os.path.join(DATA_DIR, "semantics")

# Ensure directories exist
if not os.path.isdir(RGB_DIR) or not os.path.isdir(SEM_DIR):
    raise FileNotFoundError(f"RGB or semantics directory not found under {DATA_DIR}")

# Mapping of class IDs to class names
CLASS_MAPPING = {0: "Pedestrian", 1: "Traffic Light", 2: "Car", 3: "Bus"}

# Debug: inspect unique class and instance IDs across all instance segmentation masks
mask_files = [f for f in os.listdir(SEM_DIR) if f.lower().endswith('.npy')]
unique_class_ids_all = set()
unique_instance_ids_all = set()
non_empty_count = 0
for fname in mask_files:
    mask = np.load(os.path.join(SEM_DIR, fname))  # Shape: (height, width, 3)
    class_ids = mask[:, :, 0]  # R channel for class IDs
    instance_ids = (mask[:, :, 1].astype(np.uint16) << 8) + mask[:, :, 2]  # G << 8 + B for instance IDs
    unique_class_ids = set(np.unique(class_ids))
    unique_instance_ids = set(np.unique(instance_ids))
    unique_class_ids_all.update(unique_class_ids)
    unique_instance_ids_all.update(unique_instance_ids)
    if len(unique_instance_ids - {0}) > 0:
        non_empty_count += 1
print(f"Debug: Total mask files = {len(mask_files)}, Non-empty masks = {non_empty_count}")
print(f"Debug: All unique class IDs across masks = {sorted(unique_class_ids_all)}")
print(f"Debug: All unique instance IDs across masks = {sorted(unique_instance_ids_all)}")
print("If class IDs don't match CLASS_MAPPING keys, adjust CLASS_MAPPING or mask generation.")

# 1) Count number of samples per weather condition
weather_counts = defaultdict(int)
for fname in os.listdir(RGB_DIR):
    if fname.lower().startswith("rgb_") and fname.lower().endswith(('.png', '.jpg', '.jpeg')):
        parts = fname.split("_")
        weather = "_".join(parts[1:-1])
        weather_counts[weather] += 1

# Convert to DataFrame
df_weather = pd.DataFrame(sorted(weather_counts.items()), columns=["Weather", "NumSamples"])

# 2) Count object instances per class across all instance segmentation masks
instance_counts = defaultdict(int)
for fname in mask_files:
    mask_path = os.path.join(SEM_DIR, fname)
    mask = np.load(mask_path)  # Shape: (height, width, 3)
    class_ids = mask[:, :, 0]  # R channel for class IDs
    instance_ids = (mask[:, :, 1].astype(np.uint16) << 8) + mask[:, :, 2]  # G << 8 + B for instance IDs
    unique_instances = np.unique(instance_ids)
    for inst_id in unique_instances:
        if inst_id == 0:  # Skip background
            continue
        inst_mask = (instance_ids == inst_id)
        if inst_mask.any():
            cls_id = class_ids[inst_mask][0]
            if cls_id in CLASS_MAPPING:
                instance_counts[CLASS_MAPPING[cls_id]] += 1

# Convert to DataFrame
df_instances = pd.DataFrame(sorted(instance_counts.items()), columns=["Class", "NumInstances"])

# 3) Total annotated instances
total_instances = df_instances["NumInstances"].sum() if not df_instances.empty else 0

# 4) Print summary
print("\n=== Dataset Analysis Report ===")
print("Samples per Weather Condition:")
print(df_weather.to_string(index=False))
print("\nInstances per Class:")
if df_instances.empty:
    print("No instances found. Check CLASS_MAPPING IDs and mask generation.")
else:
    print(df_instances.to_string(index=False))
print(f"\nTotal Annotated Instances: {total_instances}\n")

# 5) Visualizations
plt.figure(figsize=(8, 5))
plt.bar(df_weather['Weather'], df_weather['NumSamples'])
plt.title('Number of Samples per Weather Condition')
plt.xlabel('Weather')
plt.ylabel('Num Samples')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
if not df_instances.empty:
    plt.bar(df_instances['Class'], df_instances['NumInstances'])
    plt.title('Total Object Instances per Class')
    plt.xlabel('Class')
    plt.ylabel('Num Instances')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print('Skipping instance plot as no instances were found.')